import setup_path
import airsim
import time
import threading
import math
from pynput import keyboard

class FixedDroneController:
    def __init__(self, client=None):
        # 连接到AirSim
        print("正在连接到AirSim...")
        self.client = client if client is not None else airsim.MultirotorClient()
        self.client.confirmConnection()
        self.client.enableApiControl(True)
        self.client.armDisarm(True)
        print("连接成功!")
        
        # 控制参数
        self.max_velocity = 8.0  # 最大水平速度 (m/s)
        self.max_yaw_rate = 90.0  # 最大偏航速率 (度/秒)
        self.max_vertical_velocity = 3.0  # 最大垂直速度 (m/s)
        
        # 当前控制参数
        self.current_velocity = 2.0  # 当前水平速度 (m/s)
        self.current_yaw_rate = 30.0  # 当前偏航速率 (度/秒)
        self.current_vertical_velocity = 1.5  # 当前垂直速度 (m/s)
        
        # 控制步长
        self.velocity_step = 0.5
        self.yaw_step = 5.0
        self.vertical_step = 0.3
        
        # 控制状态
        self.is_flying = False
        self.exit_flag = False
        
        # 控制模式
        self.control_mode = "velocity"  # "velocity" 或 "position"
        
        # 按键状态
        self.pressed_keys = set()
        
        # 当前高度跟踪
        self.current_height = 0.0
        self.target_height = 10.0  # 默认目标高度 10米
        
        # 调试标志
        self.debug_mode = False
        
        print("="*70)
        print("AirSim 无人机相对朝向控制 (修复版)")
        print("="*70)
        self.print_controls()
        print("="*70)
    
    def print_controls(self):
        """打印控制说明"""
        print("飞行控制:")
        print("  [T] - 起飞到10米       [L] - 降落           [R] - 重置")
        print("  [M] - 切换模式 (速度/位置)  [D] - 切换调试模式")
        print("\n相对朝向控制 (基于无人机当前朝向):")
        print("  [W/S] - 向前/向后 (相对无人机前方)")
        print("  [A/D] - 向左/向右 (相对无人机右侧)")
        print("  [Q/E] - 左转/右转 (偏航)")
        print("  [↑/↓] - 上升/下降 (垂直运动)")
        print("\n速度调节:")
        print("  [1/2] - 增减水平速度 (当前: {:.1f} m/s)".format(self.current_velocity))
        print("  [3/4] - 增减偏航速度 (当前: {:.1f} °/s)".format(self.current_yaw_rate))
        print("  [5/6] - 增减垂直速度 (当前: {:.1f} m/s)".format(self.current_vertical_velocity))
        print("\n高级功能:")
        print("  [空格] - 急停/悬停")
        print("  [P] - 打印详细状态")
        print("  [I] - 显示无人机朝向")
        print("  [ESC] - 退出程序")
        print("\n提示: 上升下降现在应该正常工作!")
    
    def get_current_yaw(self):
        """获取当前偏航角（弧度）"""
        try:
            state = self.client.getMultirotorState()
            orientation = state.kinematics_estimated.orientation
            
            # 四元数转换为欧拉角
            pitch, roll, yaw = airsim.utils.to_eularian_angles(orientation)
            
            return yaw
        except Exception as e:
            if self.debug_mode:
                print(f"获取偏航角错误: {e}")
            return 0.0
    
    def get_current_position(self):
        """获取当前位置"""
        try:
            state = self.client.getMultirotorState()
            return state.kinematics_estimated.position
        except Exception as e:
            if self.debug_mode:
                print(f"获取位置错误: {e}")
            return airsim.Vector3r(0, 0, 0)
    
    def update_current_height(self):
        """更新当前高度"""
        try:
            pos = self.get_current_position()
            # AirSim中Z向下为正，所以高度是-Z
            self.current_height = -pos.z_val
        except:
            pass
    
    def rotate_vector_to_world(self, forward_vel, right_vel):
        """将相对向量旋转到世界坐标系（仅水平方向）"""
        yaw = self.get_current_yaw()
        
        # 创建旋转矩阵 (绕Z轴旋转)
        cos_yaw = math.cos(yaw)
        sin_yaw = math.sin(yaw)
        
        # 将相对坐标转换到世界坐标
        world_x = forward_vel * cos_yaw - right_vel * sin_yaw
        world_y = forward_vel * sin_yaw + right_vel * cos_yaw
        
        return world_x, world_y
    
    def takeoff(self, altitude=10, settle_seconds=1.0):
        """起飞到指定高度"""
        if not self.is_flying:
            print(f"起飞到 {altitude} 米高度...")
            
            # 使用takeoffAsync起飞
            self.client.takeoffAsync().join()
            print("起飞完成")
            
            # 等待稳定
            time.sleep(settle_seconds)
            
            # 使用moveToZAsync精确控制高度
            self.client.moveToZAsync(-altitude, 2).join()  # 注意：Z是负值
            print(f"已上升到 {altitude} 米")
            
            self.is_flying = True
            self.target_height = altitude
            self.update_current_height()
            print("起飞成功!")
    
    def land(self):
        """降落"""
        if self.is_flying:
            print("开始降落...")
            
            # 使用landAsync降落
            self.client.landAsync().join()
            
            self.is_flying = False
            self.current_height = 0.0
            print("已降落!")
    
    def reset(self):
        """重置"""
        print("重置位置...")
        self.client.reset()
        time.sleep(2)
        self.client.enableApiControl(True)
        self.client.armDisarm(True)
        self.is_flying = False
        self.current_height = 0.0
        print("位置已重置!")
    
    def emergency_stop(self):
        """紧急停止"""
        print("紧急停止!")
        self.client.hoverAsync().join()
        print("已悬停")
    
    def move_relative_velocity(self):
        """相对速度控制模式 - 修复版本"""
        if not self.is_flying:
            return
        
        try:
            # 获取当前位置
            current_pos = self.get_current_position()
            
            # 初始化速度
            forward_vel = 0.0
            right_vel = 0.0
            vertical_vel = 0.0  # Z轴速度
            yaw_rate = 0.0
            
            # 根据按键设置速度
            if 'w' in self.pressed_keys:  # 向前
                forward_vel = self.current_velocity
            if 's' in self.pressed_keys:  # 向后
                forward_vel = -self.current_velocity
            if 'a' in self.pressed_keys:  # 向左
                right_vel = -self.current_velocity
            if 'd' in self.pressed_keys:  # 向右
                right_vel = self.current_velocity
            if 'q' in self.pressed_keys:  # 左转
                yaw_rate = -self.current_yaw_rate
            if 'e' in self.pressed_keys:  # 右转
                yaw_rate = self.current_yaw_rate
            
            # 关键修复：正确处理上升下降
            if 'up' in self.pressed_keys:  # 上升
                # 上升：需要减小Z值（负速度），因为AirSim中Z向下为正
                vertical_vel = -self.current_vertical_velocity
                self.target_height = self.current_height + self.current_vertical_velocity * 0.1
                if self.debug_mode:
                    print(f"上升: vertical_vel={vertical_vel}, 目标高度={self.target_height}")
            
            if 'down' in self.pressed_keys:  # 下降
                # 下降：需要增加Z值（正速度）
                vertical_vel = self.current_vertical_velocity
                self.target_height = self.current_height - self.current_vertical_velocity * 0.1
                if self.debug_mode:
                    print(f"下降: vertical_vel={vertical_vel}, 目标高度={self.target_height}")
            
            # 将水平速度转换为世界坐标系
            world_x, world_y = self.rotate_vector_to_world(forward_vel, right_vel)
            
            # 调试信息
            if self.debug_mode and (forward_vel != 0 or right_vel != 0):
                print(f"相对速度: 前={forward_vel:.1f}, 右={right_vel:.1f}")
                print(f"世界速度: X={world_x:.1f}, Y={world_y:.1f}")
            
            # 方法1: 使用moveByVelocityAsync控制所有三个方向的速度
            # 这是最直接的方法，可以同时控制水平和垂直速度
            self.client.moveByVelocityAsync(
                world_x,  # X轴速度
                world_y,  # Y轴速度
                vertical_vel,  # Z轴速度（上升为负，下降为正）
                0.1,  # 持续时间
                drivetrain=airsim.DrivetrainType.MaxDegreeOfFreedom,
                yaw_mode=airsim.YawMode(is_rate=True, yaw_or_rate=yaw_rate)
            )
            
            # 更新当前高度
            self.update_current_height()
            
        except Exception as e:
            if self.debug_mode:
                print(f"移动控制错误: {e}")
    
    def adjust_horizontal_velocity(self, delta):
        """调整水平速度"""
        self.current_velocity = max(0.1, min(self.max_velocity, self.current_velocity + delta))
        print(f"水平速度: {self.current_velocity:.1f} m/s")
    
    def adjust_yaw_rate(self, delta):
        """调整偏航速率"""
        self.current_yaw_rate = max(1.0, min(self.max_yaw_rate, self.current_yaw_rate + delta))
        print(f"偏航速率: {self.current_yaw_rate:.1f} °/s")
    
    def adjust_vertical_velocity(self, delta):
        """调整垂直速度"""
        self.current_vertical_velocity = max(0.1, min(self.max_vertical_velocity, 
                                                     self.current_vertical_velocity + delta))
        print(f"垂直速度: {self.current_vertical_velocity:.1f} m/s")
    
    def toggle_control_mode(self):
        """切换控制模式"""
        if self.control_mode == "velocity":
            self.control_mode = "position"
            print("切换到位置控制模式")
        else:
            self.control_mode = "velocity"
            print("切换到速度控制模式")
    
    def toggle_debug_mode(self):
        """切换调试模式"""
        self.debug_mode = not self.debug_mode
        print(f"调试模式: {'开启' if self.debug_mode else '关闭'}")
    
    def show_drone_orientation(self):
        """显示无人机朝向"""
        if self.is_flying:
            yaw = self.get_current_yaw()
            yaw_deg = math.degrees(yaw)
            
            # 转换为0-360度范围
            if yaw_deg < 0:
                yaw_deg += 360
            
            # 显示基本方位
            direction = "未知"
            if 315 <= yaw_deg or yaw_deg < 45:
                direction = "北"
            elif 45 <= yaw_deg < 135:
                direction = "东"
            elif 135 <= yaw_deg < 225:
                direction = "南"
            elif 225 <= yaw_deg < 315:
                direction = "西"
            
            print(f"无人机朝向: {yaw_deg:.1f}° ({direction})")
        else:
            print("无人机未起飞")
    
    def print_detailed_status(self):
        """打印详细状态"""
        if self.is_flying:
            # 获取状态
            state = self.client.getMultirotorState()
            pos = state.kinematics_estimated.position
            vel = state.kinematics_estimated.linear_velocity
            
            # 获取朝向
            yaw = self.get_current_yaw()
            yaw_deg = math.degrees(yaw)
            if yaw_deg < 0:
                yaw_deg += 360
            
            print("\n" + "="*70)
            print("无人机详细状态:")
            print(f"  位置: X={pos.x_val:7.2f}m, Y={pos.y_val:7.2f}m, 高度={-pos.z_val:7.2f}m")
            print(f"  速度: X={vel.x_val:7.2f}m/s, Y={vel.y_val:7.2f}m/s, Z={vel.z_val:7.2f}m/s")
            print(f"  朝向: {yaw_deg:7.1f}°")
            print(f"  控制模式: {self.control_mode}")
            print(f"  控制参数:")
            print(f"    水平速度: {self.current_velocity:.1f} m/s")
            print(f"    垂直速度: {self.current_vertical_velocity:.1f} m/s")
            print(f"    偏航速率: {self.current_yaw_rate:.1f} °/s")
            print("="*70)
        else:
            print("无人机未起飞")
    
    def on_press(self, key):
        """按键按下事件"""
        try:
            if hasattr(key, 'char') and key.char:
                key_char = key.char.lower()
                self.pressed_keys.add(key_char)
                
                # 特殊功能键
                if key_char == 't':
                    self.takeoff()
                elif key_char == 'l':
                    self.land()
                elif key_char == 'r':
                    self.reset()
                elif key_char == ' ':
                    self.emergency_stop()
                elif key_char == 'm':
                    self.toggle_control_mode()
                elif key_char == 'd':
                    self.toggle_debug_mode()
                elif key_char == 'p':
                    self.print_detailed_status()
                elif key_char == 'i':
                    self.show_drone_orientation()
                
                # 速度调整
                elif key_char == '1':
                    self.adjust_horizontal_velocity(-self.velocity_step)
                elif key_char == '2':
                    self.adjust_horizontal_velocity(self.velocity_step)
                elif key_char == '3':
                    self.adjust_yaw_rate(-self.yaw_step)
                elif key_char == '4':
                    self.adjust_yaw_rate(self.yaw_step)
                elif key_char == '5':
                    self.adjust_vertical_velocity(-self.vertical_step)
                elif key_char == '6':
                    self.adjust_vertical_velocity(self.vertical_step)
            
            # 特殊键（方向键）
            elif key == keyboard.Key.up:
                self.pressed_keys.add('up')
                if self.debug_mode:
                    print("上箭头按下")
            elif key == keyboard.Key.down:
                self.pressed_keys.add('down')
                if self.debug_mode:
                    print("下箭头按下")
            elif key == keyboard.Key.esc:
                self.exit_flag = True
                return False
        
        except AttributeError:
            pass
    
    def on_release(self, key):
        """按键释放事件"""
        try:
            if hasattr(key, 'char') and key.char:
                key_char = key.char.lower()
                self.pressed_keys.discard(key_char)
            elif key == keyboard.Key.up:
                self.pressed_keys.discard('up')
                if self.debug_mode:
                    print("上箭头释放")
            elif key == keyboard.Key.down:
                self.pressed_keys.discard('down')
                if self.debug_mode:
                    print("下箭头释放")
        except AttributeError:
            pass
    
    def control_loop(self):
        """控制循环"""
        print("开始控制循环...按ESC退出")
        
        last_status_time = time.time()
        last_height_check = time.time()
        
        while not self.exit_flag:
            try:
                # 更新当前高度
                current_time = time.time()
                if current_time - last_height_check > 0.2:  # 每0.2秒更新一次高度
                    self.update_current_height()
                    last_height_check = current_time
                
                # 执行控制
                if self.control_mode == "velocity" and self.is_flying:
                    self.move_relative_velocity()
                
                # 定期显示状态
                if current_time - last_status_time > 0.3:  # 每0.3秒更新一次显示
                    if self.is_flying:
                        yaw = self.get_current_yaw()
                        yaw_deg = math.degrees(yaw)
                        if yaw_deg < 0:
                            yaw_deg += 360
                        
                        # 显示状态信息
                        status = f"高度: {self.current_height:5.1f}m | "
                        status += f"朝向: {yaw_deg:6.1f}° | "
                        status += f"模式: {self.control_mode:8s} | "
                        status += f"水平速度: {self.current_velocity:4.1f}m/s | "
                        status += f"垂直速度: {self.current_vertical_velocity:4.1f}m/s"
                        
                        # 显示当前按下的键
                        if self.pressed_keys:
                            active_keys = []
                            if 'w' in self.pressed_keys:
                                active_keys.append("前")
                            if 's' in self.pressed_keys:
                                active_keys.append("后")
                            if 'a' in self.pressed_keys:
                                active_keys.append("左")
                            if 'd' in self.pressed_keys:
                                active_keys.append("右")
                            if 'q' in self.pressed_keys:
                                active_keys.append("左转")
                            if 'e' in self.pressed_keys:
                                active_keys.append("右转")
                            if 'up' in self.pressed_keys:
                                active_keys.append("↑")
                            if 'down' in self.pressed_keys:
                                active_keys.append("↓")
                            
                            if active_keys:
                                status += f" | 控制: {','.join(active_keys)}"
                        
                        print(f"\r{status}", end="")
                    
                    last_status_time = current_time
                
                time.sleep(0.02)  # 50Hz控制频率
            
            except Exception as e:
                print(f"\n控制循环错误: {e}")
                time.sleep(0.1)
    
    def run(self):
        """运行控制器"""
        try:
            # 启动键盘监听
            listener = keyboard.Listener(
                on_press=self.on_press,
                on_release=self.on_release)
            listener.start()
            
            print("键盘监听已启动")
            print("提示: 确保AirSim窗口是活动窗口，或者键盘焦点在控制台窗口")
            
            # 启动控制循环
            self.control_loop()
        
        except KeyboardInterrupt:
            print("\n用户中断")
        except Exception as e:
            print(f"程序错误: {e}")
            import traceback
            traceback.print_exc()
        finally:
            self.cleanup()
    
    def cleanup(self):
        """清理资源"""
        print("\n清理资源...")
        self.exit_flag = True
        
        # 确保降落
        if self.is_flying:
            try:
                print("正在降落...")
                self.land()
            except:
                print("降落失败，尝试紧急停止...")
                try:
                    self.client.hoverAsync().join()
                except:
                    pass
        
        # 断开连接
        try:
            self.client.armDisarm(False)
            self.client.enableApiControl(False)
        except:
            pass
        
        print("程序结束")

# 测试函数
def test_vertical_control():
    """测试垂直控制"""
    print("测试垂直控制...")
    
    # 创建控制器但不启动键盘监听
    controller = FixedDroneController()
    
    try:
        # 直接控制测试
        print("1. 起飞测试")
        controller.takeoff(5)
        time.sleep(2)
        
        print("2. 上升测试")
        controller.client.moveByVelocityAsync(0, 0, -1, 2).join()  # 上升2秒
        time.sleep(1)
        
        print("3. 下降测试")
        controller.client.moveByVelocityAsync(0, 0, 1, 2).join()  # 下降2秒
        time.sleep(1)
        
        print("4. 降落")
        controller.land()
        
        print("垂直控制测试完成!")
        
    except Exception as e:
        print(f"测试错误: {e}")
    finally:
        controller.cleanup()

def main():
    """主函数"""
    import sys
    
    # 检查是否要运行测试
    if len(sys.argv) > 1 and sys.argv[1] == "--test":
        test_vertical_control()
    else:
        controller = FixedDroneController()
        controller.run()

if __name__ == "__main__":
    main()
