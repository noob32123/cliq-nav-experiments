"""
AirSim client for AerialVLA — captures front + down camera views.
Modified from CityNavAgent version to grab dual-view images at 256x256.
"""
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
try:
    import setup_path
except ImportError:
    pass

import airsim
import time
import numpy as np
import cv2


class AerialVLAAirSimClient:
    """AirSim wrapper for AerialVLA's dual-view (front + down) capture."""

    def __init__(self, image_dir=None):
        print("[AerialVLA] Connecting to AirSim...")
        self.client = airsim.MultirotorClient()
        self.client.confirmConnection()
        self.client.enableApiControl(True)
        self.client.armDisarm(True)
        print("[AerialVLA] Connected.")

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

    def takeoff(self, altitude=10.0):
        print(f"[AerialVLA] Takeoff → {altitude}m")
        self.client.takeoffAsync().join()
        time.sleep(1.0)
        self.client.moveToZAsync(-altitude, 2).join()

    def land(self):
        self.client.landAsync().join()
        time.sleep(2.0)

    def get_pose(self):
        return self.client.simGetVehiclePose()

    def get_position(self):
        pose = self.client.simGetVehiclePose()
        return np.array([pose.position.x_val, pose.position.y_val, pose.position.z_val])

    def get_orientation_quat(self):
        pose = self.client.simGetVehiclePose()
        o = pose.orientation
        return np.array([o.w_val, o.x_val, o.y_val, o.z_val])

    def capture_dual_view(self):
        """Capture front (Camera 0) + down (Camera 3) RGB images."""
        responses = self.client.simGetImages([
            airsim.ImageRequest("0", airsim.ImageType.Scene, False, False),
            airsim.ImageRequest("3", airsim.ImageType.Scene, False, False),
        ])
        images = {}

        for idx, resp in enumerate(responses):
            if not resp or resp.height == 0:
                continue
            data = resp.image_data_uint8
            h, w = resp.height, resp.width
            if len(data) == h * w * 3:
                img = np.frombuffer(data, dtype=np.uint8).reshape(h, w, 3)
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                if idx == 0:
                    images['front'] = img
                elif idx == 1:
                    images['down'] = img

        if self._image_dir:
            for name, img in images.items():
                cv2.imwrite(os.path.join(self._image_dir, f"{name}.png"),
                            cv2.cvtColor(img, cv2.COLOR_RGB2BGR))

        return images
