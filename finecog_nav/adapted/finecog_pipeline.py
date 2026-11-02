"""
FineCog-Nav Pipeline — orchestrates the 8 cognitive modules per step.
"""
import time
import numpy as np


class FineCogPipeline:

    ACTION_NAMES = {0: "STOP", 1: "MOVE_FORWARD", 2: "TURN_LEFT", 3: "TURN_RIGHT",
                    4: "ASCEND", 5: "DESCEND", 6: "MOVE_LEFT", 7: "MOVE_RIGHT"}

    def __init__(self, modules, flight_ctrl, airsim_adapter, perception_module):
        self.modules = modules
        self.flight = flight_ctrl
        self.airsim = airsim_adapter
        self.perception = perception_module
        self.step_count = 0
        self.max_steps = 20
        self.no_progress_count = 0

    def run_step(self):
        """
        Execute one full step of FineCog-Nav pipeline.
        Returns True to continue, False if TASK_FINISH (action 0) selected.
        """
        step = self.step_count
        print(f"\n{'='*50}")
        print(f"[FineCog] Step {step + 1}/{self.max_steps}")
        print(f"  Subgoal [{self.modules.current_subgoal_idx+1}/{len(self.modules.subgoals or [1])}]: "
              f"{self.modules.current_subgoal_text()}")
        print(f"{'='*50}")

        # 1. Capture images
        images = self.perception.capture_all()
        rgb = images.get('front_rgb')
        depth_vis = images.get('front_depth')

        if rgb is None:
            print("[FineCog] Image capture failed, retrying...")
            time.sleep(1)
            images = self.perception.capture_all()
            rgb = images.get('front_rgb')
            depth_vis = images.get('front_depth')

        # 2. Perception (VLM)
        print("[FineCog] 2. Perception (VLM)...")
        perception_output = self.modules.perception(rgb) if rgb is not None else "(no image)"
        print(f"  Perception: {perception_output[:150]}...")

        # 3. Extract subgoals if needed
        if not self.modules.subgoals:
            print("[FineCog] 3. Subgoal Extractor...")
            self.modules.subgoal_extractor(perception_output)
            print(f"  Subgoals: {self.modules.subgoals}")
            if not self.modules.subgoals:
                print("[FineCog] No subgoals generated, using instruction as subgoal")
                cur = self.modules.parsed_instructions[self.modules.current_instruction_idx] if self.modules.current_instruction_idx < len(self.modules.parsed_instructions) else None
                self.modules.subgoals = [cur["sub_instruction"]] if cur else ["navigate to target"]

        # 4. Subgoal Judger
        print("[FineCog] 4. Subgoal Judger...")
        achieved, reason = self.modules.subgoal_judger(perception_output)
        print(f"  Achieved: {achieved}, Reason: {reason[:120]}")

        if achieved and self.modules.has_more_subgoals():
            print("[FineCog] Subgoal completed! Advancing...")
            has_more = self.modules.advance_subgoal()
            if not has_more:
                print("[FineCog] All subgoals and instructions completed. Finishing.")
                self.flight.client.hoverAsync().join()
                return False
            print(f"[FineCog] New subgoal: {self.modules.current_subgoal_text()}")
            self.no_progress_count = 0

        # 5. Decision-Making
        print("[FineCog] 5. Decision-Making...")
        action_code, thought = self.modules.decision_making(perception_output, depth_vis)
        action_name = self.ACTION_NAMES.get(action_code, f"UNKNOWN({action_code})")
        print(f"  Decision: {action_name} | Thought: {thought[:120]}")

        # 6. Step Memory
        print("[FineCog] 6. Step Memory...")
        self.modules.step_memory(perception_output, action_code)

        # 7. Execute action
        if action_code == 0:
            print("[FineCog] Navigation complete (TASK_FINISH).")
            self.flight.client.hoverAsync().join()
            return False

        pos_before = self.flight.get_position()
        self.flight.execute_discrete_action(action_code)
        pos_after = self.flight.get_position()
        time.sleep(2.0)

        dist = np.linalg.norm(pos_after[:2] - pos_before[:2])
        if dist < 0.1 and action_code not in (2, 3):
            self.no_progress_count += 1
            if self.no_progress_count >= 3:
                print("[FineCog] No progress for 3 steps with non-turn actions. Finishing.")
                self.flight.client.hoverAsync().join()
                return False
        else:
            self.no_progress_count = 0

        self.step_count += 1
        if self.step_count >= self.max_steps:
            print("[FineCog] Max steps reached.")
            self.flight.client.hoverAsync().join()
            return False

        return True

    def run_navigation_full(self, instruction_text, goal_position, success_threshold=10.0):
        """Full navigation run with all FineCog-Nav modules.
        (reset_to_start, takeoff, initial_turn, and landing are handled externally)"""
        self.step_count = 0
        self.no_progress_count = 0

        # One-time: Instruction Parser
        print(f"\n[FineCog] 1. Instruction Parser...")
        self.modules.instruction_parser(instruction_text)
        print(f"  Parsed {len(self.modules.parsed_instructions)} instruction sentences")

        for _ in range(self.max_steps):
            should_continue = self.run_step()
            if not should_continue:
                break

        final_pos = self.flight.get_position()
        goal_2d = np.array(goal_position[:2])
        nav_error = float(np.linalg.norm(final_pos[:2] - goal_2d))
        success = nav_error <= success_threshold

        return {
            'success': success,
            'nav_error': nav_error,
            'steps': self.step_count,
            'trajectory': self.flight.get_trajectory(),
            'subgoal_memory': self.modules.subgoal_memory,
            'instruction_memory': self.modules.instruction_memory,
        }

    def _exec_initial_turn(self, degrees):
        import airsim
        yaw_rate = 30.0
        value = abs(degrees)
        duration = max(0.5, value / yaw_rate)
        rate = yaw_rate if degrees > 0 else -yaw_rate
        print(f"[FineCog] Initial turn: {degrees:.1f} deg")
        self.airsim.client.moveByVelocityAsync(
            0, 0, 0, duration,
            yaw_mode=airsim.YawMode(is_rate=True, yaw_or_rate=rate)
        ).join()
        time.sleep(4.0)
