# -*- coding: utf-8 -*-
"""
生成十四条航路的GNSS拒止前历史航迹。
要求：
- 针对现有的14个场景，分别生成一个 history_track.csv
- 每个文件包含6个历史航迹点 (P_{-6} 到 P_{-1})，即 H0 到 H5
- 使得这六个点每个点之间共五个向量加权和，严格平行于实际的 VLM 平面航向
- 每个场景的侧向弯曲幅度和方向不同，使得转向夹角完全唯一且各不相同
"""

import os
import argparse
import csv
import math
import numpy as np
import random

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# SCENARIO_CONFIG 包含目标等，但这里主要用它来获取每个场景的名字
SCENARIO_CONFIG = {
    "data_1":  {"name": "orangewhite_umbrella"},
    "data_2":  {"name": "statue"},
    "data_3":  {"name": "restaurant_across_river"},
    "data_4":  {"name": "hotel_front_door"},
    "data_5":  {"name": "blue_amusement_park"},
    "data_6":  {"name": "fountain"},
    "data_7":  {"name": "house"},
    "data_8":  {"name": "playground"},
    "data_9":  {"name": "parkbench"},
    "data_10": {"name": "umbrella"},
    "data_11": {"name": "house2"},
    "data_12": {"name": "fountain2"},
    "data_13": {"name": "sculpture"},
    "data_14": {"name": "sculpture2"},
}

def get_initial_heading_points(csv_path, xy_threshold=0.5):
    points = []
    with open(csv_path, 'r', encoding='utf-8-sig') as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                points.append(np.array([float(row['X']), float(row['Y']), float(row['Z'])]))
            except (ValueError, KeyError):
                continue
                
    if len(points) < 2:
        return None, None
        
    for i in range(len(points) - 1):
        p_curr = points[i]
        p_next = points[i+1]
        
        # 计算平面 XY 距离变化
        xy_dist = math.hypot(p_next[0] - p_curr[0], p_next[1] - p_curr[1])
        
        # 如果 XY 平面变化显著，说明开始朝着目标方向移动了
        if xy_dist >= xy_threshold:
            return p_curr, p_next
            
    # 如果找不到显著的 XY 变化，Fallback 回前两个点
    return points[0], points[1]

def main(history_dir=None):
    history_dir = history_dir or os.path.join(ROOT, "history_tracks")
    os.makedirs(history_dir, exist_ok=True)
    
    # 权重设定 (离现在起始点越近权重越高)
    weights = [0.1, 0.15, 0.2, 0.25, 0.3]
    print(f"使用的历史航向权重 (从远到近): {weights}, 总和: {sum(weights):.2f}")
    
    for data_key in sorted(SCENARIO_CONFIG.keys(), key=lambda k: int(k.split('_')[1])):
        config = SCENARIO_CONFIG[data_key]
        scenario_name = config["name"]
        data_dir = os.path.join(ROOT, data_key)
        
        clip_dir = os.path.join(data_dir, f"qwen3.5-35b-a3b_CLIP_{scenario_name}")
        if not os.path.isdir(clip_dir):
            print(f"未找到文件夹: {clip_dir}")
            continue
            
        csv_files = sorted([f for f in os.listdir(clip_dir) if f.endswith('.csv')])
        if not csv_files:
            continue
            
        p0, p1 = None, None
        for csv_file in csv_files:
            csv_path = os.path.join(clip_dir, csv_file)
            curr_p0, curr_p1 = get_initial_heading_points(csv_path, xy_threshold=0.5)
            # 检查是否有显著的平面位移
            if curr_p0 is not None and curr_p1 is not None:
                if math.hypot(curr_p1[0] - curr_p0[0], curr_p1[1] - curr_p0[1]) >= 0.5:
                    p0, p1 = curr_p0, curr_p1
                    break
                    
        # 如果所有文件都没有显著平面位移，就退回到使用第一个文件的原Fallback结果
        if p0 is None or p1 is None:
            p0, p1 = get_initial_heading_points(os.path.join(clip_dir, csv_files[0]), xy_threshold=0.5)
            
        if p0 is None or p1 is None:
            print(f"{data_key}: 轨迹点不足2个，无法计算方向")
            continue
        
        # 实际航向向量
        v_actual = p1 - p0
        norm_v = np.linalg.norm(v_actual)
        if norm_v < 1e-5:
            print(f"{data_key}: P0和P1太近，无法确定方向")
            continue
            
        v_dir = v_actual / norm_v
        
        # 为了让航迹变得曲折(曲线)，我们在 v_dir 的垂直方向(XY平面)引入偏移量
        if abs(v_dir[0]) < 1e-3 and abs(v_dir[1]) < 1e-3:
            n_dir = np.array([1.0, 0.0, 0.0])
        else:
            n_dir = np.array([-v_dir[1], v_dir[0], 0.0])
            n_dir = n_dir / np.linalg.norm(n_dir)
            
        # 设计横向偏移分量 d_raw (产生一个平滑的S弯或C弯)
        # 索引 0~4 对应 V1~V5
        d_raw = np.array([2.0, 1.0, 0.0, -1.0, -2.0])
        
        # 严格的数学保证：必须满足 sum(w_i * d_i) = 0
        w_array = np.array(weights)
        mu = np.sum(w_array * d_raw) / np.sum(w_array)
        d_adjusted = d_raw - mu  # 调整后的偏移使得其加权和严格为0
        
        # 使用随机数赋予不同场景随机的弯曲幅度和方向，使各个场景转角随机且各不相同
        base_amplitude = random.uniform(0.5, 3.0)
        direction_sign = random.choice([1.0, -1.0])
        curve_amplitude = base_amplitude * direction_sign
        
        d_final = d_adjusted * curve_amplitude
        
        # 基础前进纵向步长
        step_size = 3.0
        
        # 计算 5 个向量 V1 到 V5
        vectors = []
        for i in range(5):
            v_i = step_size * v_dir + d_final[i] * n_dir
            vectors.append(v_i)
            
        # 计算连接 P0 的向量 V6 (可以直接往前直走)
        v6 = step_size * v_dir
        
        # 反向倒推历史点坐标
        # H5 是最靠近 P0 的点
        h5 = p0 - v6
        h4 = h5 - vectors[4]
        h3 = h4 - vectors[3]
        h2 = h3 - vectors[2]
        h1 = h2 - vectors[1]
        h0 = h1 - vectors[0]
        
        history_points = [h0, h1, h2, h3, h4, h5]
        
        # 验证计算
        v1 = history_points[1] - history_points[0]
        v2 = history_points[2] - history_points[1]
        v3 = history_points[3] - history_points[2]
        v4 = history_points[4] - history_points[3]
        v5 = history_points[5] - history_points[4]
        
        weighted_sum = weights[0]*v1 + weights[1]*v2 + weights[2]*v3 + weights[3]*v4 + weights[4]*v5
        cos_check = np.dot(weighted_sum, v_actual) / (np.linalg.norm(weighted_sum) * norm_v)
        print(f"{data_key} ({scenario_name}):")
        print(f"  P0 = [{p0[0]:.2f}, {p0[1]:.2f}, {p0[2]:.2f}]")
        print(f"  P1 = [{p1[0]:.2f}, {p1[1]:.2f}, {p1[2]:.2f}]")
        print(f"  实际 V = [{v_actual[0]:.2f}, {v_actual[1]:.2f}, {v_actual[2]:.2f}]")
        print(f"  振幅 = {curve_amplitude:.2f}, 验证 Cos(theta) = {cos_check:.6f}")
        
        # 写入历史文件
        out_csv = os.path.join(history_dir, f"{data_key}_history.csv")
        with open(out_csv, 'w', newline='', encoding='utf-8-sig') as f:
            writer = csv.writer(f)
            writer.writerow(["Point_Index", "X", "Y", "Z", "Description"])
            for i, p in enumerate(history_points):
                writer.writerow([i, p[0], p[1], p[2], f"History_{i-6}"])
                
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--history-dir", default=os.path.join(ROOT, "history_tracks"))
    args = parser.parse_args()
    main(args.history_dir)
