import sys
import torch
import clip
from PIL import Image
import setup_path
import os
import time
import re
import base64
import numpy as np
import cv2
from openai import OpenAI
import airsim
import csv
from fly_on_keyboard import FixedDroneController  # 复用键盘控制脚本中的类


_model_cache = {}

# 编号对应的位置描述（行优先，从上到下，从左到右）
POSITION_DESC = {
    1: "前上方偏左",
    2: "正前上方",
    3: "前上方偏右",
    4: "前方偏左",
    5: "正前方",
    6: "前方偏右",
    7: "前下方偏左",
    8: "正前下方",
    9: "前下方偏右"
}

def reset_to_start(client, start_position, start_orientation):
    """
    将无人机重置到指定的起点。
    :param client: AirSim 客户端
    :param start_position: 元组 (x, y, z)，NED坐标系（z负值表示高度）
    :param start_orientation: 四元数 (w, x, y, z)，默认悬停朝向
    """
    # 1. 重置整个仿真到初始状态（清除物理效果、恢复车辆）
    client.reset()
    
    # 2. 重新获取API控制并解锁（reset后需要重新启用）
    client.enableApiControl(True)
    client.armDisarm(True)
    
    # # 3. 构造目标位姿
    # if start_orientation is None:
    #     # 默认朝北（单位四元数）
    #     start_orientation = airsim.Quaternionr(1, 0, 0, 0)
    # else:
    #     start_orientation = airsim.Quaternionr(*start_orientation)
    orientation = airsim.Quaternionr(*start_orientation)

    target_pose = airsim.Pose(
        airsim.Vector3r(*start_position),
        orientation
    )
    
    # 4. 将无人机传送到指定位置（忽略碰撞，直接放置）
    client.simSetVehiclePose(target_pose, ignore_collision=True)
    
    # 5. 可选：等待一小段时间让仿真稳定
    import time
    time.sleep(2.0)
    
    # 验证位置
    pose = client.simGetVehiclePose()
    print(f"已重置到起点: X={pose.position.x_val:.2f}, Y={pose.position.y_val:.2f}, Z={pose.position.z_val:.2f}")

# ========== CLIP函数 ==========
def find_best_patch(image_path, text, model_name="ViT-B/32", device="cuda:0", save_dir=None):
    """
    将输入图片分割成3x3九个小块，找出与文本最匹配的小块，并可选择保存所有小块。

    Args:
        image_path (str): 输入图片路径
        text (str): 文本描述，例如 "a dog"
        model_name (str): CLIP模型名称，默认 "ViT-B/32"
        device (str): 运行设备，默认 "cuda:0"
        save_dir (str, optional): 保存小图片的目录，默认为None（不保存）。如果指定，则将九个小块保存到该目录。

    Returns:
        best_index (int): 最匹配的小块编号，从1到9
        best_logit (float): 对应的logits值（越大表示越匹配）
        best_patch (PIL.Image): 最匹配的小块图像
        position (str): 该小块在原图中的位置描述，例如 "上方偏左"
    """
    # 设备设置
    device = torch.device(device if torch.cuda.is_available() and "cuda" in device else "cpu")

    # 加载模型（带缓存）
    cache_key = (model_name, device)
    if cache_key not in _model_cache:
        print(f"Loading CLIP model '{model_name}' on {device}...")
        model, preprocess = clip.load(model_name, device=device)
        _model_cache[cache_key] = (model, preprocess)
    else:
        model, preprocess = _model_cache[cache_key]

    # 打开图片，调整尺寸至能被3整除
    try:
        img = Image.open(image_path).convert("RGB")
    except Exception as e:
        raise ValueError(f"无法打开图片 {image_path}: {e}")

    w, h = img.size
    new_w = (w // 3) * 3
    new_h = (h // 3) * 3
    if new_w != w or new_h != h:
        img = img.resize((new_w, new_h), Image.Resampling.LANCZOS)

    tile_w, tile_h = new_w // 3, new_h // 3

    # 准备保存目录（如果需要保存）
    if save_dir is not None:
        os.makedirs(save_dir, exist_ok=True)
        base_name = os.path.splitext(os.path.basename(image_path))[0]

    # 分割九个小块并预处理
    patches = []
    image_tensors = []
    for row in range(3):
        for col in range(3):
            left = col * tile_w
            upper = row * tile_h
            right = left + tile_w
            lower = upper + tile_h
            patch = img.crop((left, upper, right, lower))
            patches.append(patch)
            tensor = preprocess(patch).unsqueeze(0)      # (1,3,224,224)
            image_tensors.append(tensor)

            # 如果需要保存，则保存当前小块
            if save_dir is not None:
                patch_index = row * 3 + col + 1
                save_path = os.path.join(save_dir, f"{base_name}_part_{patch_index}.png")
                patch.save(save_path)
                # print(f"已保存: {save_path}")

    images_batch = torch.cat(image_tensors, dim=0).to(device)  # (9,3,224,224)
    text_tokens = clip.tokenize([text]).to(device)

    # 推理
    with torch.no_grad():
        logits_per_image, _ = model(images_batch, text_tokens)  # (9,1)
        logits = logits_per_image.squeeze(-1).cpu().numpy()     # (9,)

    # 找出最佳小块
    best_idx = logits.argmax()          # 0~8
    best_index = best_idx + 1            # 转为1~9
    best_logit = float(logits[best_idx])
    best_patch = patches[best_idx]
    position = POSITION_DESC[best_index]

    return best_index, best_logit, best_patch, position

# ========== 图片编码函数 ==========
def encode_numpy_to_data_url(img_np, image_format='jpeg'):
    """
    将 numpy 数组编码为 base64 data URL
    支持灰度图 (H,W) 或彩色图 (H,W,3)
    """
    success, buffer = cv2.imencode(f'.{image_format}', img_np)
    if not success:
        raise RuntimeError("图像编码失败")
    b64_str = base64.b64encode(buffer).decode('utf-8')
    return f"data:image/{image_format};base64,{b64_str}"

# ========== 动作解析 ==========
def parse_action(action_str):
    """
    从模型输出中提取动作和数值
    期望格式如: forward 2, turn_left 30, stop 等
    返回 (动作名称, 数值) 或 (None, None) 若解析失败
    """
    action_str = action_str.strip().lower()
    if action_str == 'stop':
        return 'stop', 0

    pattern = r'^(forward|backward|left|right|up|down|turn_left|turn_right)\s+(\d+(?:\.\d+)?)$'
    match = re.match(pattern, action_str)
    if match:
        action = match.group(1)
        value = float(match.group(2))
        return action, value
    else:
        print(f"警告：无法解析动作指令 '{action_str}'")
        return None, None

# ========== 动作执行 ==========
def execute_action(controller, action, value):
    """
    根据动作指令控制无人机移动
    """
    horizontal_speed = controller.current_velocity
    vertical_speed = controller.current_vertical_velocity
    yaw_rate = controller.current_yaw_rate
    client = controller.client

    if action in ['forward', 'backward', 'left', 'right']:
        duration = max(0.5, value / horizontal_speed)
        if action == 'forward':
            forward_vel = horizontal_speed
            right_vel = 0
        elif action == 'backward':
            forward_vel = -horizontal_speed
            right_vel = 0
        elif action == 'left':
            forward_vel = 0
            right_vel = -horizontal_speed
        elif action == 'right':
            forward_vel = 0
            right_vel = horizontal_speed
        else:
            return
        # 相对速度转世界坐标
        vx, vy = controller.rotate_vector_to_world(forward_vel, right_vel)
        print(f"执行 {action} {value:.1f} 米 (速度 {horizontal_speed:.1f} m/s, 持续时间 {duration:.1f}s)")
        client.moveByVelocityAsync(vx, vy, 0, duration).join()

    elif action in ['up', 'down']:
        duration = max(0.5, value / vertical_speed)
        if action == 'up':
            vz = -vertical_speed   # 上升为负
        else:
            vz = vertical_speed    # 下降为正
        print(f"执行 {action} {value:.1f} 米 (速度 {vertical_speed:.1f} m/s, 持续时间 {duration:.1f}s)")
        client.moveByVelocityAsync(0, 0, vz, duration).join()

    elif action in ['turn_left', 'turn_right']:
        duration = max(0.5, value / yaw_rate)
        if action == 'turn_left':
            rate = -yaw_rate
        else:
            rate = yaw_rate
        print(f"执行 {action} {value:.1f} 度 (速率 {yaw_rate:.1f} °/s, 持续时间 {duration:.1f}s)")
        client.moveByVelocityAsync(0, 0, 0, duration,
                                   yaw_mode=airsim.YawMode(is_rate=True, yaw_or_rate=rate)).join()
    else:
        print(f"未知动作: {action}")

# ========== 主流程 ==========
def main():
    # 初始化 OpenAI 客户端（需设置环境变量 DASHSCOPE_API_KEY）
    client_ai = OpenAI(
        api_key=os.getenv("DASHSCOPE_API_KEY"),
        base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
    )

    # 初始化无人机控制器（复用键盘控制脚本的类）
    drone = FixedDroneController()
    print("无人机已连接。")

    # init status
    orientation = airsim.Quaternionr(0.9236459732055664, 0, 0, -0.3832468092441559)
    start_position = (-253.011, -115.011, -10.1460)
    reset_to_start(drone.client, start_position, orientation)
    altitude = 1  # 米
    drone.takeoff(altitude)
    execute_action(drone, 'turn_right', 160)
    # execute_action(drone, 'forward', 30)

    with open('flight_log.csv', 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(["Timestamp", "X", "Y", "Z"])
        pose = drone.client.simGetVehiclePose()
        writer.writerow([time.time(), pose.position.x_val, pose.position.y_val, pose.position.z_val])

    # # 起飞到指定高度
    # altitude = 1  # 米
    # drone.takeoff(altitude)
    # print(f"已起飞至 {altitude} 米。")

    # # 目标提示（可根据需要修改）
    # target_prompt = (
    #     "你是一架无人机的飞手，现在正在依靠无人机上搭载的相机飞行。当前任务：请智能地导航到前方灰白色房子的门口附近并降落，不必太过精细，大概到达目标附近后即可输出stop"
    #     "我会按顺序给你三张无人机上搭载的相机拍到的图片：前向图、前向深度图、俯视图。请依据深度图像的信息尽量避免碰撞并灵活的导航到目标附近后降落。"
    #     "请根据这些图片决定下一步动作，你可以采取的动作如下列所示。"
    #     "输出格式必须为以下之一（数值为数字）：\n"
    #     "- forward X（向前飞X米）\n"
    #     "- backward X（向后飞X米）\n"
    #     "- left X（向左飞X米）\n"
    #     "- right X（向右飞X米）\n"
    #     "- up X（上升X米）\n"
    #     "- down X（下降X米）\n"
    #     "- turn_left Y（左转Y度）\n"
    #     "- turn_right Y（右转Y度）\n"
    #     "- stop（到达目标，停止导航）\n"
    #     "只输出一条指令，不要附加任何解释。"
    # )

    # 定义三个图像请求：前向彩色、前向深度、俯视彩色
    requests = [
        airsim.ImageRequest("0", airsim.ImageType.Scene, False, False),      # 前向彩色
        airsim.ImageRequest("0", airsim.ImageType.DepthVis, False, False),   # 前向深度可视化
        airsim.ImageRequest("3", airsim.ImageType.Scene, False, False)       # 俯视彩色
    ]

    max_iterations = 25
    for i in range(max_iterations):
        print(f"\n===== 迭代 {i+1} =====")

        # 1. 获取三张图像
        responses = drone.client.simGetImages(requests)

        # 2. 将响应转换为 data URL 列表
        image_urls = []
        for idx, resp in enumerate(responses):
            if not resp or resp.height == 0:
                print(f"警告：第 {idx+1} 张图像获取失败")
                # 如果某图像失败，可以跳过（但会破坏输入数量），这里直接退出本次迭代
                # 更稳健的做法是重试，但为简洁，跳过本次迭代
                continue

            h, w = resp.height, resp.width
            data = resp.image_data_uint8

            # 根据数据长度判断单通道（深度）或三通道（彩色）
            if len(data) == h * w:           # 单通道灰度
                img_gray = np.frombuffer(data, dtype=np.uint8).reshape(h, w)
                img_rgb = cv2.cvtColor(img_gray, cv2.COLOR_GRAY2RGB)   # 转为三通道
                cv2.imwrite(f"{idx}.png", img_rgb)
            elif len(data) == h * w * 3:     # 三通道彩色
                img_rgb = np.frombuffer(data, dtype=np.uint8).reshape(h, w, 3)
                cv2.imwrite(f"{idx}.png", img_rgb)
            else:
                print(f"警告：图像数据尺寸异常，跳过")
                continue

            data_url = encode_numpy_to_data_url(img_rgb)
            image_urls.append({"type": "image_url", "image_url": {"url": data_url}})

        # 如果获取的图像不足三张，重试或结束
        if len(image_urls) < 3:
            print("未能获取三张完整图像，等待后重试...")
            time.sleep(2)
            continue
        
        # 目标提示（可根据需要修改）
        target_prompt = (
        "你是一架无人机的飞手，现在正在依靠无人机上搭载的相机飞行。当前任务：请智能地导航到前方橘白色条纹遮阳伞的附近并降落，不必太过精细，大概到达目标附近后即可输出stop"
        "我会按顺序给你三张无人机上搭载的相机拍到的图片：前向图、前向深度图、俯视图。请依据深度图像的信息尽量避免碰撞附近的物体并灵活的导航到目标附近后降落。"
        "请根据这些图片决定下一步动作，你可以采取的动作如下列所示。"
        "输出格式必须为以下之一（数值为数字）：\n"
        "- forward X（向前飞X米）\n"
        "- backward X（向后飞X米）\n"
        "- left X（向左飞X米）\n"
        "- right X（向右飞X米）\n"
        "- up X（上升X米）\n"
        "- down X（下降X米）\n"
        "- turn_left Y（左转Y度）\n"
        "- turn_right Y（右转Y度）\n"
        "- stop（到达目标，停止导航）\n"
        "只输出一条指令，不要附加任何解释。"
        )

        # 对图片调用clip进行语义匹配
        # idx, score, patch_img, pos = find_best_patch("0.png", "orange and white striped umbrella", save_dir="./frontcam_seg")
        # target_prompt = target_prompt + f'为辅助你的判断：你的导航目标有很大的可能性在你的{pos}'
        # print(pos)
        # print(target_prompt)

        # 3. 构造消息
        messages = [
            {
                "role": "user",
                "content": image_urls + [{"type": "text", "text": target_prompt}]
            }
        ]

        # 4. 调用大模型
        print("正在请求大模型决策...")
        try:
            completion = client_ai.chat.completions.create(
                model="qwen3.5-35b-a3b",
                messages=messages,
                timeout=150
            )
            action_str = completion.choices[0].message.content
            print(f"模型输出: {action_str}")
        except Exception as e:
            print(f"模型调用失败: {e}")
            break
        
        # action_str = 'forward 1'

        # 5. 解析动作
        action, value = parse_action(action_str)
        if action is None:
            print("无法解析动作，跳过本次迭代")
            time.sleep(1)
            continue

        # 6. 检查是否停止
        if action == 'stop':
            print("模型指示已到达目标，导航结束。")
            break

        # 7. 执行动作
        execute_action(drone, action, value)

        # 等待稳定
        time.sleep(4.0)
        

        with open('flight_log.csv', 'a', newline='') as file:
            writer = csv.writer(file)
            pose = drone.client.simGetVehiclePose()
            writer.writerow([time.time(), pose.position.x_val, pose.position.y_val, pose.position.z_val])

    # 导航结束，降落
    print("导航结束，准备降落...")
    drone.land()
    drone.cleanup()
    print("任务完成。")

    with open('flight_log.csv', 'a', newline='') as file:
        writer = csv.writer(file)
        pose = drone.client.simGetVehiclePose()
        writer.writerow([time.time(), pose.position.x_val, pose.position.y_val, pose.position.z_val])



if __name__ == "__main__":
    main()