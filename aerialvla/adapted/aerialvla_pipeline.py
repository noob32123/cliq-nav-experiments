"""
AerialVLA Pipeline — per-step loop:
  1. Capture front + down RGB
  2. Compute fuzzy direction from target position
  3. Build prompt + model inference
  4. Parse action + landing check
  5. Execute action via flight controller
"""
import time
import numpy as np


class AerialVLAPipeline:

    def __init__(self, model_wrapper, flight_ctrl, airsim_client, prompt_builder, direction_fn):
        self.model = model_wrapper
        self.flight = flight_ctrl
        self.airsim = airsim_client
        self.build_prompt = prompt_builder
        self.compute_direction = direction_fn
        self.step_count = 0
        self.max_steps = 20

    def run_step(self, object_description, goal_position):
        """One step of AerialVLA pipeline. Returns False if landing."""
        print(f"\n[AerialVLA] Step {self.step_count + 1}/{self.max_steps}")

        # 1. Capture dual view
        images = self.airsim.capture_dual_view()
        front_rgb = images.get('front')
        down_rgb = images.get('down')
        if front_rgb is None or down_rgb is None:
            print("[AerialVLA] Image capture failed, retrying...")
            time.sleep(1)
            images = self.airsim.capture_dual_view()
            front_rgb = images.get('front')
            down_rgb = images.get('down')

        # 2. Directional prompt
        drone_pos = self.flight.get_position()
        drone_quat = self.flight.get_orientation_quat()
        direction = self.compute_direction(drone_pos, drone_quat, goal_position)
        prompt = self.build_prompt(direction, object_description)
        print(f"  Direction: '{direction}'")

        # 3. Model inference
        result = self.model.run(front_rgb, down_rgb, prompt)
        fwd, down, yaw_rate = result['fwd'], result['down'], result['yaw']
        land = result['land']
        print(f"  Action: fwd={fwd:.2f} down={down:.2f} yaw={yaw_rate:.2f} LAND={land}")

        # 4. Check landing
        if land:
            print("[AerialVLA] Landing signal detected.")
            return False

        # 5. Execute
        self.flight.execute_continuous_action(fwd, down, yaw_rate)
        self.step_count += 1

        if self.step_count >= self.max_steps:
            print("[AerialVLA] Max steps reached.")
            return False

        return True

    def run_navigation(self, object_description, goal_position):
        self.step_count = 0
        for _ in range(self.max_steps):
            should_continue = self.run_step(object_description, goal_position)
            if not should_continue:
                break
