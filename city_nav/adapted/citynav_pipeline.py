"""
CityNavAgent-style exploration pipeline adapted for this environment.
Implements the core decision loop:

1. Capture observations (front RGB, front depth, top-down views)
2. Send to Qwen VLM with CityNavAgent-style structured prompt
3. Parse LLM output into discrete action code (0-7)
4. Execute action via FlightController
5. Repeat until STOP or max iterations
"""
import time
import re
import json
import numpy as np


class CityNavPipeline:
    """
    Adapted CityNavAgent pipeline.
    Uses LLM-only for perception + planning (no DINO/SAM grounder).
    Action space: 8 discrete actions with fixed step sizes.
    """

    ACTION_CODES = {
        "STOP": 0, "MOVE_FORWARD": 1, "TURN_LEFT": 2,
        "TURN_RIGHT": 3, "GO_UP": 4, "GO_DOWN": 5,
        "MOVE_LEFT": 6, "MOVE_RIGHT": 7
    }

    ACTION_NAMES = {v: k for k, v in ACTION_CODES.items()}

    def __init__(self, airsim_adapter, flight_ctrl, qwen_llm, perception):
        self.airsim = airsim_adapter
        self.flight = flight_ctrl
        self.llm = qwen_llm
        self.perception = perception

        self.step_count = 0
        self.max_steps = 20
        self.action_history = []

    def build_citynav_prompt(self, instruction_text, target_description):
        """
        Build a CityNavAgent-style structured prompt.
        This mimics the original CityNavAgent's landmark-based hierarchical planning
        but adapts it to this environment's 3-camera setup.

        Key design choices matching CityNavAgent:
        - Uses "subgoals" concept: the target as the sole subgoal
        - Uses discrete action space (8 actions, fixed step sizes)
        - Requests structured output for parsing
        """
        action_list_str = ", ".join([
            "STOP(0)", "MOVE_FORWARD(1)", "TURN_LEFT(2)", "TURN_RIGHT(3)",
            "GO_UP(4)", "GO_DOWN(5)", "MOVE_LEFT(6)", "MOVE_RIGHT(7)"
        ])

        prev_actions = ""
        if self.action_history:
            prev_actions = "Previous actions: " + ", ".join(self.action_history[-5:])

        prompt = f"""You are an AI agent controlling a drone to complete a navigation task.

Navigation Instruction: {instruction_text}
Target to find: {target_description}

You have 3 camera views provided above:
- Image 1: Front-facing RGB camera
- Image 2: Front-facing depth visualization (brighter = closer objects)
- Image 3: Top-down camera

Your task: based on these observations, decide the NEXT discrete action to move toward the target.

Action Space (output ONLY the action name from this list):
{action_list_str}

Step sizes:
- MOVE_FORWARD/MOVE_BACKWARD: 5 meters
- MOVE_LEFT/MOVE_RIGHT: 5 meters
- TURN_LEFT/TURN_RIGHT: 15 degrees
- GO_UP/GO_DOWN: 2 meters

{prev_actions}

Output format - output ONLY one line with the action name:
Action: <ACTION_NAME>
Reason: <brief explanation of why>

Examples:
Action: MOVE_FORWARD
Reason: target area is visible ahead, approaching

Action: TURN_LEFT
Reason: scanning for the target, turning to get better view

Action: GO_UP
Reason: need higher altitude for better overview

Action: STOP
Reason: target reached, navigation complete

Now, based on the 3 images above, what is your next action?"""

        return prompt

    def parse_action_response(self, response_text):
        """
        Parse LLM response into a discrete action code.
        Matches CityNavAgent's action_parser logic.
        """
        response_text = response_text.strip().lower()

        match = re.search(r'action:\s*(\w+)', response_text)
        if match:
            action_str = match.group(1).strip().upper()
            if action_str in self.ACTION_CODES:
                return self.ACTION_CODES[action_str], response_text

        for name, code in self.ACTION_CODES.items():
            if name.lower() in response_text:
                return code, response_text

        if "stop" in response_text:
            return 0, response_text
        if "forward" in response_text or "move forward" in response_text:
            return 1, response_text
        if "turn left" in response_text or "left" in response_text[:10]:
            return 2, response_text
        if "turn right" in response_text or "right" in response_text[:10]:
            return 3, response_text
        if "go up" in response_text or "ascend" in response_text:
            return 4, response_text
        if "go down" in response_text or "descend" in response_text:
            return 5, response_text
        if "move left" in response_text:
            return 6, response_text
        if "move right" in response_text:
            return 7, response_text

        print(f"[Pipeline] Could not parse action from: {response_text[:200]}")
        return -1, response_text

    def run_step(self, instruction_text, target_description):
        """
        Execute one step of the navigation pipeline.
        Returns True if should continue, False if STOP.
        """

        print(f"\n{'='*50}")
        print(f"[Pipeline] Step {self.step_count + 1}/{self.max_steps}")
        print(f"{'='*50}")

        images = self.perception.capture_all()
        front_rgb = images.get('front_rgb')
        front_depth = images.get('front_depth')
        top_down = images.get('top_down')

        if front_rgb is None:
            print("[Pipeline] WARNING: Failed to capture images, retrying...")
            time.sleep(1.0)
            images = self.perception.capture_all()
            front_rgb = images.get('front_rgb')
            front_depth = images.get('front_depth')
            top_down = images.get('top_down')

        prompt = self.build_citynav_prompt(instruction_text, target_description)

        viewpoint_images = {}
        if front_rgb is not None:
            viewpoint_images['front_rgb'] = front_rgb
        if front_depth is not None:
            viewpoint_images['front_depth'] = front_depth
        if top_down is not None:
            viewpoint_images['top_down'] = top_down

        print("[Pipeline] Requesting LLM decision...")
        try:
            response = self.llm.query_viewpoint_api(prompt, viewpoint_images, max_tokens=200)
            print(f"[Pipeline] LLM response: {response[:300]}")
        except Exception as e:
            print(f"[Pipeline] LLM call failed: {e}")
            return True

        action_code, full_response = self.parse_action_response(response)

        if action_code < 0:
            print("[Pipeline] Could not determine action, trying default MOVE_FORWARD...")
            action_code = 1

        self.action_history.append(self.ACTION_NAMES.get(action_code, "UNKNOWN"))
        self.step_count += 1

        if action_code == 0:  # STOP
            print("[Pipeline] Navigation complete - STOP received.")
            return False

        self.flight.execute_discrete_action(action_code)
        time.sleep(2.0)

        if self.step_count >= self.max_steps:
            print("[Pipeline] Max steps reached.")
            return False

        return True

    def run_navigation(self, instruction_text, target_description):
        """
        Full navigation loop.
        """
        self.step_count = 0
        self.action_history = []

        print(f"\n{'#'*60}")
        print(f"# CityNavAgent Navigation")
        print(f"# Instruction: {instruction_text}")
        print(f"# Target: {target_description}")
        print(f"{'#'*60}")

        for _ in range(self.max_steps):
            should_continue = self.run_step(instruction_text, target_description)
            if not should_continue:
                break

        print(f"\n[Pipeline] Navigation finished after {self.step_count} steps.")
        print(f"[Pipeline] Action history: {self.action_history}")

    def run_navigation_with_goal(self, instruction_text, target_description,
                                   start_position, goal_position, altitude,
                                   initial_turn=None, success_threshold=10.0):
        """
        Full navigation loop with goal checking.
        Returns metrics compatible with the user's evaluation format.
        """
        self.flight.reset_to_start(start_position)
        self.flight.takeoff(altitude)
        time.sleep(2.0)

        if initial_turn is not None and abs(initial_turn) > 0.5:
            self.flight.execute_initial_turn(initial_turn)
            time.sleep(1.0)

        self.run_navigation(instruction_text, target_description)

        final_pos = self.flight.get_position()
        goal_pos = np.array(goal_position[:2])
        final_pos_2d = final_pos[:2]
        nav_error = np.linalg.norm(final_pos_2d - goal_pos)
        success = nav_error <= success_threshold

        trajectory = self.flight.get_trajectory()
        trajectory_length = 0.0
        if len(trajectory) > 1:
            diffs = trajectory[1:] - trajectory[:-1]
            trajectory_length = np.sum(np.linalg.norm(diffs[:, :2], axis=1))

        self.flight.land()

        metrics = {
            'success': success,
            'nav_error': nav_error,
            'trajectory_length': trajectory_length,
            'steps': self.step_count,
            'action_history': self.action_history,
            'trajectory': trajectory.tolist() if len(trajectory) > 0 else [],
            'final_position': final_pos.tolist(),
            'goal_position': goal_pos.tolist(),
        }

        return metrics

    def run_basic_validation(self, altitude=10.0, num_loops=3):
        """
        Basic validation: takeoff, fly a few steps, land.
        Does NOT use LLM - just tests AirSim integration.

        CityNavAgent's takeoff pattern: execute GO_UP x5 (5 steps of 2m = 10m)
        """
        print(f"\n{'#'*60}")
        print(f"# CityNavAgent Basic Validation")
        print(f"# Testing: Takeoff -> MOVE_FORWARD x{num_loops} -> Land")
        print(f"{'#'*60}")

        start_pos = self.flight.get_position()
        print(f"Start position: {start_pos}")

        self.flight.takeoff(altitude)
        time.sleep(1.0)

        for i in range(num_loops):
            print(f"\n--- Validation step {i+1}/{num_loops} ---")
            pos_before = self.flight.get_position()
            yaw_before = self.flight.get_yaw()
            print(f"  Before: pos={pos_before}, yaw={np.rad2deg(yaw_before):.1f}deg")

            self.flight.execute_discrete_action(1)

            pos_after = self.flight.get_position()
            yaw_after = self.flight.get_yaw()
            dist = np.linalg.norm(pos_after[:2] - pos_before[:2])
            print(f"  After:  pos={pos_after}, yaw={np.rad2deg(yaw_after):.1f}deg")
            print(f"  Distance moved: {dist:.2f}m")
            time.sleep(1.0)

        self.flight.land()

        final_pos = self.flight.get_position()
        total_dist = np.linalg.norm(final_pos[:2] - start_pos[:2])
        print(f"\nTotal horizontal displacement: {total_dist:.2f}m")
        print(f"Trajectory points: {len(self.flight.trajectory)}")

        if total_dist > 1.0:
            print("[VALIDATION SUCCESS] Drone moved. AirSim integration working.")
        else:
            print("[VALIDATION WARNING] Drone didn't move significantly. Check AirSim.")

        return self.flight.trajectory
