"""
Direct AirSim client adapter - replaces the RPC-based AirVLNSimulatorClientTool.
Uses airsim.MultirotorClient directly, matching the user's nav_qwen.py pattern.
"""
import sys
import os

_current_adapted_dir = os.path.dirname(os.path.abspath(__file__))
_project_root = os.path.abspath(os.path.join(_current_adapted_dir, '..', '..'))
_airsim_root = os.path.abspath(os.path.join(_project_root, '..', '..'))
sys.path.insert(0, _project_root)
if os.path.exists(os.path.join(_airsim_root, 'airsim', 'client.py')):
    sys.path.insert(0, _airsim_root)

try:
    import setup_path  # noqa: E402
except ImportError:
    pass

import airsim
import time
import numpy as np
import cv2
import base64


class AirSimClientAdapter:
    """
    Direct AirSim wrapper using MultirotorClient.
    Provides the same interface needed by CityNavAgent pipeline.
    """

    ACTION_CODES = {
        "STOP": 0,
        "MOVE_FORWARD": 1,
        "TURN_LEFT": 2,
        "TURN_RIGHT": 3,
        "GO_UP": 4,
        "GO_DOWN": 5,
        "MOVE_LEFT": 6,
        "MOVE_RIGHT": 7
    }

    ACTION_NAMES = {v: k for k, v in ACTION_CODES.items()}

    FORWARD_STEP = 5.0
    UP_DOWN_STEP = 2.0
    LEFT_RIGHT_STEP = 5.0
    TURN_ANGLE = 15.0
    TILT_ANGLE = 15.0

    HORIZONTAL_SPEED = 2.0
    VERTICAL_SPEED = 1.5
    YAW_RATE = 30.0

    def __init__(self, image_dir=None):
        print("[AirSimAdapter] Connecting to AirSim...")
        self.client = airsim.MultirotorClient()
        self.client.confirmConnection()
        self.client.enableApiControl(True)
        self.client.armDisarm(True)
        print("[AirSimAdapter] Connected.")

        self._image_dir = image_dir
        if self._image_dir:
            os.makedirs(self._image_dir, exist_ok=True)

        self._image_requests = [
            airsim.ImageRequest("0", airsim.ImageType.Scene, False, False),
            airsim.ImageRequest("0", airsim.ImageType.DepthVis, False, False),
            airsim.ImageRequest("3", airsim.ImageType.Scene, False, False),
        ]

    def reset_to_pose(self, position, orientation_quat):
        self.client.reset()
        time.sleep(1.0)
        self.client.enableApiControl(True)
        self.client.armDisarm(True)
        orientation = airsim.Quaternionr(*orientation_quat)
        pose = airsim.Pose(airsim.Vector3r(*position), orientation)
        self.client.simSetVehiclePose(pose, ignore_collision=True)
        time.sleep(2.0)
        current = self.client.simGetVehiclePose()
        print(f"[AirSimAdapter] Reset to: X={current.position.x_val:.2f}, "
              f"Y={current.position.y_val:.2f}, Z={current.position.z_val:.2f}")

    def takeoff(self, altitude=10.0):
        print(f"[AirSimAdapter] Taking off to {altitude}m...")
        self.client.takeoffAsync().join()
        time.sleep(1.0)
        self.client.moveToZAsync(-altitude, 2).join()
        print("[AirSimAdapter] Takeoff complete.")

    def land(self):
        print("[AirSimAdapter] Landing...")
        self.client.landAsync().join()
        time.sleep(2.0)
        print("[AirSimAdapter] Landed.")

    def get_pose(self):
        pose = self.client.simGetVehiclePose()
        return pose

    def get_position(self):
        pose = self.client.simGetVehiclePose()
        return np.array([pose.position.x_val, pose.position.y_val, pose.position.z_val])

    def get_yaw(self):
        pose = self.client.simGetVehiclePose()
        _, _, yaw = airsim.to_eularian_angles(pose.orientation)
        return yaw

    def set_pose(self, pose):
        if isinstance(pose, np.ndarray):
            pos = airsim.Vector3r(float(pose[0]), float(pose[1]), float(pose[2]))
            ori = airsim.Quaternionr(1, 0, 0, 0)
            if len(pose) >= 7:
                ori = airsim.Quaternionr(float(pose[3]), float(pose[4]),
                                         float(pose[5]), float(pose[6]))
            elif len(pose) == 6:
                import math
                yaw = float(pose[5])
                ori = airsim.to_quaternion(0, 0, yaw)
            p = airsim.Pose(pos, ori)
        elif isinstance(pose, airsim.Pose):
            p = pose
        else:
            raise ValueError(f"Unsupported pose type: {type(pose)}")

        self.client.simSetVehiclePose(p, ignore_collision=True)

    def capture_images(self, flight_label=""):
        responses = self.client.simGetImages(self._image_requests)
        images = {}
        for idx, resp in enumerate(responses):
            if not resp or resp.height == 0:
                continue
            h, w = resp.height, resp.width
            data = resp.image_data_uint8
            if len(data) == h * w:
                img = np.frombuffer(data, dtype=np.uint8).reshape(h, w)
                img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
            elif len(data) == h * w * 3:
                img = np.frombuffer(data, dtype=np.uint8).reshape(h, w, 3)
            else:
                continue
            if idx == 0:
                images['front_rgb'] = img
            elif idx == 1:
                images['front_depth'] = img
            elif idx == 2:
                images['top_down'] = img

        if self._image_dir:
            prefix = f"{flight_label}_" if flight_label else ""
            for name, img in images.items():
                path = os.path.join(self._image_dir, f"{prefix}{name}.png")
                cv2.imwrite(path, img)
        return images

    def image_to_base64(self, img_np):
        success, buffer = cv2.imencode('.jpeg', img_np)
        if not success:
            raise RuntimeError("Image encoding failed")
        b64_str = base64.b64encode(buffer).decode('utf-8')
        return f"data:image/jpeg;base64,{b64_str}"

    def execute_action(self, action_code, num_steps=1):
        if isinstance(action_code, str):
            action_code = self.ACTION_CODES.get(action_code.upper(), -1)
        if action_code < 0 or action_code > 7:
            print(f"[AirSimAdapter] Unknown action: {action_code}")
            return

        name = self.ACTION_NAMES.get(action_code, "UNKNOWN")
        client = self.client

        # CityNavAgent step sizes
        h_step = self.FORWARD_STEP
        v_step = self.UP_DOWN_STEP
        lr_step = self.LEFT_RIGHT_STEP
        turn_step = self.TURN_ANGLE

        # Use velocity-based control with appropriate speeds
        h_speed = self.HORIZONTAL_SPEED
        v_speed = self.VERTICAL_SPEED
        yaw_rate = self.YAW_RATE

        for _ in range(num_steps):
            duration = 0.5
            vx, vy, vz = 0.0, 0.0, 0.0
            yaw_mode = airsim.YawMode(is_rate=False, yaw_or_rate=0.0)

            if action_code == 1:  # MOVE_FORWARD
                duration = max(0.5, h_step / h_speed)
                yaw = self.get_yaw()
                vx = h_speed * np.cos(yaw)
                vy = h_speed * np.sin(yaw)
                print(f"  -> MOVE_FORWARD {h_step}m")

            elif action_code == 2:  # TURN_LEFT
                duration = max(0.5, turn_step / yaw_rate)
                yaw_mode = airsim.YawMode(is_rate=True, yaw_or_rate=-yaw_rate)
                print(f"  -> TURN_LEFT {turn_step}deg")

            elif action_code == 3:  # TURN_RIGHT
                duration = max(0.5, turn_step / yaw_rate)
                yaw_mode = airsim.YawMode(is_rate=True, yaw_or_rate=yaw_rate)
                print(f"  -> TURN_RIGHT {turn_step}deg")

            elif action_code == 4:  # GO_UP
                duration = max(0.5, v_step / v_speed)
                vz = -v_speed
                print(f"  -> GO_UP {v_step}m")

            elif action_code == 5:  # GO_DOWN
                duration = max(0.5, v_step / v_speed)
                vz = v_speed
                print(f"  -> GO_DOWN {v_step}m")

            elif action_code == 6:  # MOVE_LEFT
                duration = max(0.5, lr_step / h_speed)
                yaw = self.get_yaw()
                vx = -h_speed * np.sin(yaw)
                vy = h_speed * np.cos(yaw)
                print(f"  -> MOVE_LEFT {lr_step}m")

            elif action_code == 7:  # MOVE_RIGHT
                duration = max(0.5, lr_step / h_speed)
                yaw = self.get_yaw()
                vx = h_speed * np.sin(yaw)
                vy = -h_speed * np.cos(yaw)
                print(f"  -> MOVE_RIGHT {lr_step}m")

            elif action_code == 0:  # STOP
                print("  -> STOP")
                break

            client.moveByVelocityAsync(
                vx, vy, vz, duration,
                drivetrain=airsim.DrivetrainType.MaxDegreeOfFreedom,
                yaw_mode=yaw_mode
            ).join()
            time.sleep(1.0)

    def rotate_vector_to_world(self, forward_vel, right_vel):
        yaw = self.get_yaw()
        vx = forward_vel * np.cos(yaw) - right_vel * np.sin(yaw)
        vy = forward_vel * np.sin(yaw) + right_vel * np.cos(yaw)
        return vx, vy
