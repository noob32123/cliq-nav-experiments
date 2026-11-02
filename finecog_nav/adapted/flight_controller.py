"""
Flight controller adapter - wraps AirSim movement commands.
Uses EXACTLY the same API calls as nav_qwen.py for consistency.
"""
import time
import math
import numpy as np
import airsim


class FlightController:
    """
    Executes drone control using the same API patterns as nav_qwen.py.
    Turn operations use moveByVelocityAsync with yaw_mode, matching execute_action().
    """

    HORIZONTAL_SPEED = 2.0    # m/s  (matches nav_qwen FixedDroneController.current_velocity)
    VERTICAL_SPEED = 1.5      # m/s  (matches nav_qwen)
    YAW_RATE = 30.0           # deg/s (matches nav_qwen)

    def __init__(self, airsim_adapter):
        self.airsim = airsim_adapter
        self.client = airsim_adapter.client
        self.is_flying = False
        self.trajectory = []

    def reset_to_start(self, position, orientation_quat=None):
        self.airsim.reset_to_pose(position, orientation_quat or (1, 0, 0, 0))
        self.is_flying = False
        self.trajectory = []
        self._log_position()

    def takeoff(self, altitude=10.0):
        print(f"[FlightCtrl] Taking off to {altitude}m...")
        self.client.takeoffAsync().join()
        time.sleep(1.0)
        self.client.moveToZAsync(-altitude, 2).join()
        self.is_flying = True
        print("[FlightCtrl] Takeoff complete.")
        self._log_position()

    def land(self):
        print("[FlightCtrl] Landing...")
        self.client.landAsync().join()
        time.sleep(2.0)
        self.is_flying = False
        self._log_position()
        print("[FlightCtrl] Landed.")

    def get_position(self):
        return self.airsim.get_position()

    def get_yaw(self):
        return self.airsim.get_yaw()

    def _log_position(self):
        pose = self.client.simGetVehiclePose()
        pos = np.array([pose.position.x_val, pose.position.y_val, pose.position.z_val])
        self.trajectory.append(pos.copy())
        return pos

    def get_trajectory(self):
        return np.array(self.trajectory)

    def rotate_vector_to_world(self, forward_vel, right_vel):
        yaw = self.get_yaw()
        world_x = forward_vel * math.cos(yaw) - right_vel * math.sin(yaw)
        world_y = forward_vel * math.sin(yaw) + right_vel * math.cos(yaw)
        return world_x, world_y

    # ==================== IDENTICAL to nav_qwen.py execute_action ====================

    def _lock_pose(self):
        """Lock current pose via simSetVehiclePose to prevent AirSim yaw drift."""
        pose = self.client.simGetVehiclePose()
        self.client.simSetVehiclePose(pose, ignore_collision=True)
        time.sleep(0.5)

    def turn_left(self, degrees):
        """Turn left using rotateToYawAsync to absolutely lock the yaw."""
        current_yaw = self.airsim.get_yaw()
        target_yaw = current_yaw - math.radians(abs(degrees))
        target_yaw = (target_yaw + math.pi) % (2 * math.pi) - math.pi
        print(f"[FlightCtrl] turn_left {degrees:.1f} deg (target_yaw={math.degrees(target_yaw):.1f}deg)")
        self.client.rotateToYawAsync(target_yaw, 5.0, 1.0).join()
        time.sleep(1.0)
        self._log_position()

    def turn_right(self, degrees):
        """Turn right using rotateToYawAsync to absolutely lock the yaw."""
        current_yaw = self.airsim.get_yaw()
        target_yaw = current_yaw + math.radians(abs(degrees))
        target_yaw = (target_yaw + math.pi) % (2 * math.pi) - math.pi
        print(f"[FlightCtrl] turn_right {degrees:.1f} deg (target_yaw={math.degrees(target_yaw):.1f}deg)")
        self.client.rotateToYawAsync(target_yaw, 5.0, 1.0).join()
        time.sleep(1.0)
        self._log_position()

    def execute_initial_turn(self, degrees):
        """
        Execute initial turn exactly as nav_qwen.py does.
        Positive = turn_right, Negative = turn_left.
        """
        if degrees is None or abs(degrees) < 0.5:
            print("[FlightCtrl] No initial turn needed.")
            return
        if degrees > 0:
            self.turn_right(abs(degrees))
        else:
            self.turn_left(abs(degrees))

    # ==================== CityNavAgent 8 discrete actions ====================

    FORWARD_STEP = 5.0
    UP_DOWN_STEP = 2.0
    LEFT_RIGHT_STEP = 5.0
    TURN_ANGLE_DISCRETE = 15.0

    ACTION_CODES = {
        "STOP": 0, "MOVE_FORWARD": 1, "TURN_LEFT": 2,
        "TURN_RIGHT": 3, "GO_UP": 4, "GO_DOWN": 5,
        "MOVE_LEFT": 6, "MOVE_RIGHT": 7
    }
    ACTION_NAMES = {v: k for k, v in ACTION_CODES.items()}

    def execute_discrete_action(self, action_code):
        """Execute one CityNavAgent discrete action step."""
        name = self.ACTION_NAMES.get(action_code, "UNKNOWN")

        if action_code == 0:  # STOP
            self.client.hoverAsync().join()
            self._log_position()
            return

        h_speed = self.HORIZONTAL_SPEED
        v_speed = self.VERTICAL_SPEED

        vx, vy, vz = 0.0, 0.0, 0.0
        duration = 0.5
        use_yaw = False
        yaw_is_rate = True
        yaw_val = 0.0

        if action_code == 1:  # MOVE_FORWARD
            duration = max(0.5, self.FORWARD_STEP / h_speed)
            yaw = self.get_yaw()
            vx = h_speed * math.cos(yaw)
            vy = h_speed * math.sin(yaw)

        elif action_code == 2:  # TURN_LEFT (rotateToYawAsync is absolute yaw control)
            target_yaw = self.get_yaw() - math.radians(self.TURN_ANGLE_DISCRETE)
            target_yaw = (target_yaw + math.pi) % (2 * math.pi) - math.pi
            self.client.rotateToYawAsync(target_yaw, 5.0, 1.0).join()
            time.sleep(0.5)
            self._log_position()
            return

        elif action_code == 3:  # TURN_RIGHT
            target_yaw = self.get_yaw() + math.radians(self.TURN_ANGLE_DISCRETE)
            target_yaw = (target_yaw + math.pi) % (2 * math.pi) - math.pi
            self.client.rotateToYawAsync(target_yaw, 5.0, 1.0).join()
            time.sleep(0.5)
            self._log_position()
            return

        elif action_code == 4:  # GO_UP
            duration = max(0.5, self.UP_DOWN_STEP / v_speed)
            vz = -v_speed

        elif action_code == 5:  # GO_DOWN
            duration = max(0.5, self.UP_DOWN_STEP / v_speed)
            vz = v_speed

        elif action_code == 6:  # MOVE_LEFT
            duration = max(0.5, self.LEFT_RIGHT_STEP / h_speed)
            yaw = self.get_yaw()
            vx = -h_speed * math.sin(yaw)
            vy = h_speed * math.cos(yaw)

        elif action_code == 7:  # MOVE_RIGHT
            duration = max(0.5, self.LEFT_RIGHT_STEP / h_speed)
            yaw = self.get_yaw()
            vx = h_speed * math.sin(yaw)
            vy = -h_speed * math.cos(yaw)

        if use_yaw:
            self.client.moveByVelocityAsync(
                vx, vy, vz, duration,
                drivetrain=airsim.DrivetrainType.MaxDegreeOfFreedom,
                yaw_mode=airsim.YawMode(is_rate=yaw_is_rate, yaw_or_rate=yaw_val)
            ).join()
        else:
            self.client.moveByVelocityAsync(vx, vy, vz, duration).join()
        time.sleep(0.5)
        self._log_position()
