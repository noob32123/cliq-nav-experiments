"""
AerialVLA Flight Controller — translates continuous 3-DoF actions to AirSim.
"""
import time
import math
import numpy as np
import airsim


class AerialVLAFlightController:
    """Executes AerialVLA's continuous (fwd, down, yaw_rate) actions."""

    CRUISE_SPEED = 1.0
    YAW_THRESHOLD = 0.25

    def __init__(self, airsim_adapter):
        self.airsim = airsim_adapter
        self.client = airsim_adapter.client
        self.is_flying = False
        self.trajectory = []
        self.total_yaw = 0.0

    def reset_to_start(self, position, orientation_quat=None):
        self.airsim.reset_to_pose(position, orientation_quat or (1, 0, 0, 0))
        self.is_flying = False
        self.trajectory = []
        self.total_yaw = 0.0
        self._log_position()

    def takeoff(self, altitude=10.0):
        print(f"[AerialVLA] Takeoff → {altitude}m")
        self.client.takeoffAsync().join()
        time.sleep(1.0)
        self.client.moveToZAsync(-altitude, 2).join()
        self.is_flying = True
        self._log_position()

    def land(self):
        self.client.landAsync().join()
        time.sleep(2.0)
        self.is_flying = False
        self._log_position()

    def get_position(self):
        return self.airsim.get_position()

    def get_orientation_quat(self):
        return self.airsim.get_orientation_quat()

    def get_current_yaw(self):
        pose = self.airsim.get_pose()
        _, _, yaw = airsim.to_eularian_angles(pose.orientation)
        return yaw

    def _log_position(self):
        pose = self.client.simGetVehiclePose()
        pos = np.array([pose.position.x_val, pose.position.y_val, pose.position.z_val])
        self.trajectory.append(pos.copy())
        return pos

    def get_trajectory(self):
        return np.array(self.trajectory)

    def execute_initial_turn(self, degrees):
        """Turn at start using nav_qwen.py method."""
        if abs(degrees) < 0.5:
            return
        yaw_rate = 30.0
        duration = max(0.5, abs(degrees) / yaw_rate)
        rate = yaw_rate if degrees > 0 else -yaw_rate
        self.client.moveByVelocityAsync(
            0, 0, 0, duration,
            yaw_mode=airsim.YawMode(is_rate=True, yaw_or_rate=rate)
        ).join()
        time.sleep(4.0)
        self._log_position()

    def execute_continuous_action(self, fwd, down, yaw_rate):
        """
        Execute AerialVLA output: forward velocity, down (vertical) velocity, yaw rate.
        All in body frame. fwd in m/s, down in m/s (+ = descend), yaw_rate in rad/s (+ = right).
        """

        curr_pos = self.get_position()
        curr_yaw = self.get_current_yaw()

        target_yaw = curr_yaw + yaw_rate
        target_yaw = (target_yaw + math.pi) % (2 * math.pi) - math.pi
        target_yaw_deg = math.degrees(target_yaw)

        # Compute world-frame displacement
        if abs(yaw_rate) < self.YAW_THRESHOLD:
            # Small yaw: move forward while gradually adjusting yaw
            total_dist = math.sqrt(fwd**2 + down**2)
            if total_dist < 1.0:
                dx = fwd * math.cos(target_yaw)
                dy = fwd * math.sin(target_yaw)
                vx, vy, vz = dx / 1.0, dy / 1.0, down / 1.0
                dur = 1.0
            else:
                dx = fwd * math.cos(target_yaw)
                dy = fwd * math.sin(target_yaw)
                vx = (dx / total_dist) * self.CRUISE_SPEED
                vy = (dy / total_dist) * self.CRUISE_SPEED
                vz = (down / total_dist) * self.CRUISE_SPEED
                dur = total_dist / self.CRUISE_SPEED

            self.client.rotateToYawAsync(target_yaw_deg, timeout_sec=3, margin=5).join()
            self.client.moveByVelocityAsync(
                vx, vy, vz, duration=dur,
                drivetrain=airsim.DrivetrainType.MaxDegreeOfFreedom,
                yaw_mode=airsim.YawMode(is_rate=False, yaw_or_rate=target_yaw_deg)
            ).join()

        else:
            # Large yaw: turn first, then adjust altitude
            self.client.rotateToYawAsync(target_yaw_deg, timeout_sec=3, margin=5).join()
            target_z = curr_pos[2] + down
            self.client.moveToZAsync(float(target_z), velocity=2.0, timeout_sec=3).join()

        # Stabilization hover
        self.client.moveByVelocityAsync(0, 0, 0, duration=0.5).join()
        time.sleep(0.5)
        self._log_position()
