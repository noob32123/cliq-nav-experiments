"""
Perception module - adapted camera capture for CityNavAgent.
Replaces 5-view panorama (via drone rotation) with fixed 3-camera setup:
- Camera "0": Front RGB + Front DepthVis
- Camera "3": Top-down RGB

Also generates synthetic multi-view by rotating the drone.
"""
import time
import numpy as np
import cv2


class PerceptionModule:
    """
    Handles image capture and processing for the CityNavAgent pipeline.
    Matches the user's existing 3-camera setup.
    """

    OBSERVATION_DIRECTIONS = ["left", "slightly_left", "front", "slightly_right", "right"]
    YAW_OFFSETS = {
        "left": -85,
        "slightly_left": -35,
        "front": 0,
        "slightly_right": 35,
        "right": 85,
    }

    def __init__(self, airsim_adapter):
        self.airsim = airsim_adapter
        self.client = airsim_adapter.client

    def capture_front_rgb(self):
        images = self.airsim.capture_images()
        return images.get('front_rgb')

    def capture_front_depth(self):
        images = self.airsim.capture_images()
        return images.get('front_depth')

    def capture_top_down(self):
        images = self.airsim.capture_images()
        return images.get('top_down')

    def capture_all(self):
        return self.airsim.capture_images()

    def get_pano_by_rotation(self):
        """
        Generate 5-view panorama by rotating the drone in place.
        Matches CityNavAgent's approach but using direct control.
        WARNING: This physically rotates the drone!
        """
        import airsim

        observations = {}
        original_yaw_rate = self.airsim.YAW_RATE

        for direction in self.OBSERVATION_DIRECTIONS:
            yaw_offset = self.YAW_OFFSETS[direction]
            print(f"  [Perception] Rotating to capture {direction} view ({yaw_offset}deg)...")

            if yaw_offset != 0:
                rot_duration = abs(yaw_offset) / self.airsim.YAW_RATE
                yaw_dir = np.sign(yaw_offset) * self.airsim.YAW_RATE
                self.client.moveByVelocityAsync(
                    0, 0, 0, max(0.5, rot_duration),
                    yaw_mode=airsim.YawMode(is_rate=True, yaw_or_rate=yaw_dir)
                ).join()
                time.sleep(1.0)

            images = self.airsim.capture_images()
            observations[direction] = {
                'rgb': images.get('front_rgb'),
                'depth': images.get('front_depth'),
                'top_down': images.get('top_down')
            }

            if yaw_offset != 0:
                rot_duration = abs(yaw_offset) / self.airsim.YAW_RATE
                yaw_dir = -np.sign(yaw_offset) * self.airsim.YAW_RATE
                self.client.moveByVelocityAsync(
                    0, 0, 0, max(0.5, rot_duration),
                    yaw_mode=airsim.YawMode(is_rate=True, yaw_or_rate=yaw_dir)
                ).join()
                time.sleep(1.0)

        return observations

    def collect_viewpoints(self, direction_map=None):
        """
        Collect images for each viewpoint direction.
        If direction_map is provided as {direction: image_path}, use those.
        Otherwise returns the captured images keyed by direction.
        """
        if direction_map:
            return direction_map

        images = self.airsim.capture_images()
        return {
            "front": images,
            "left": images,
            "right": images,
            "slightly_left": images,
            "slightly_right": images
        }

    def get_image_for_direction(self, direction="front"):
        """
        Get the primary (front) image. Other directions are not physically available
        without rotation, so we return the front image as a sensible default.
        """
        images = self.airsim.capture_images()
        return {
            'rgb': images.get('front_rgb'),
            'depth': images.get('front_depth'),
            'top_down': images.get('top_down')
        }
