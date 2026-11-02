"""
FineCog-Nav 8 Cognitive Modules
================================
Each module is a function: takes structured inputs, calls LLM/VLM, returns dict.
All prompts are loaded from finecog_nav/prompts/*.txt
"""
import os
import json

_PROMPT_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "prompts")


def _load_prompt(name):
    with open(os.path.join(_PROMPT_DIR, name), "r", encoding="utf-8") as f:
        return f.read()


class FineCogModules:
    """Wraps all 8 FineCog-Nav cognitive modules with state management."""

    def __init__(self, llm_client, flight_ctrl, collision_checker, airsim_adapter):
        self.llm = llm_client
        self.flight = flight_ctrl
        self.collision = collision_checker
        self.airsim = airsim_adapter

        self.parsed_instructions = []
        self.current_instruction_idx = 0
        self.subgoals = []
        self.current_subgoal_idx = 0

        self.attention_cache = {}
        self.imagination_cache = {}

        self.step_memories = []
        self.subgoal_memory = ""
        self.instruction_memory = ""

    # ─── Module 1: Instruction Parser ───

    def instruction_parser(self, instruction):
        prompt = _load_prompt("instruction_parser.txt").format(instruction=instruction)
        response = self.llm.query_text(prompt)
        parsed = self.llm.extract_json(response)
        if parsed and isinstance(parsed, list):
            self.parsed_instructions = parsed
        else:
            self.parsed_instructions = [{"sub_instruction": instruction, "landmarks": []}]
        self.current_instruction_idx = 0
        self.subgoals = []
        self.current_subgoal_idx = 0
        return self.parsed_instructions

    # ─── Module 2: Subgoal Extractor ───

    def subgoal_extractor(self, perception_output):
        if self.current_instruction_idx >= len(self.parsed_instructions):
            return []
        cur = self.parsed_instructions[self.current_instruction_idx]
        landmarks = ", ".join(cur.get("landmarks", []))
        prompt = _load_prompt("subgoal_extractor.txt").format(
            current_instruction=cur["sub_instruction"],
            landmarks=landmarks,
            perception_output=perception_output
        )
        response = self.llm.query_text(prompt)
        parsed = self.llm.extract_json(response)
        if parsed and isinstance(parsed, list):
            self.subgoals = parsed
            self.current_subgoal_idx = 0
        else:
            self.subgoals = [cur["sub_instruction"]]
            self.current_subgoal_idx = 0
        return self.subgoals

    # ─── Module 3: Attention ───

    def attention(self):
        if self.current_instruction_idx >= len(self.parsed_instructions):
            return []
        cur = self.parsed_instructions[self.current_instruction_idx]
        cur_landmarks = ", ".join(cur.get("landmarks", []))
        next_landmarks = ""
        if self.current_instruction_idx + 1 < len(self.parsed_instructions):
            nxt = self.parsed_instructions[self.current_instruction_idx + 1]
            next_landmarks = ", ".join(nxt.get("landmarks", []))
        cache_key = (self.current_instruction_idx, cur_landmarks, next_landmarks)
        if cache_key in self.attention_cache:
            return self.attention_cache[cache_key]
        prompt = _load_prompt("attention.txt").format(
            current_landmarks=cur_landmarks,
            next_landmarks=next_landmarks
        )
        response = self.llm.query_text(prompt)
        parsed = self.llm.extract_json(response)
        if parsed and isinstance(parsed, list):
            self.attention_cache[cache_key] = parsed
            return parsed
        self.attention_cache[cache_key] = []
        return []

    # ─── Module 4: Perception (VLM - uses image) ───

    def perception(self, rgb_image):
        attention_qs = self.attention()
        qs_text = "\n".join(f"- {q.get('landmark', '?')}: {q.get('question', '?')}"
                            for q in attention_qs) if attention_qs else "No specific landmarks to focus on."
        prompt = _load_prompt("perception.txt").format(attention_questions=qs_text)
        response = self.llm.query_vision(prompt, rgb_image)
        parsed = self.llm.extract_json(response)
        if parsed and isinstance(parsed, dict):
            perception_text = parsed.get("overall", "") + "\n" + parsed.get("details", "")
            return perception_text
        return response or "No visual observation available."

    # ─── Module 5: Imagination ───

    def imagination(self):
        if self.current_instruction_idx >= len(self.parsed_instructions):
            return ""
        cur = self.parsed_instructions[self.current_instruction_idx]
        landmarks = ", ".join(cur.get("landmarks", []))
        subgoal = (self.subgoals[self.current_subgoal_idx]
                    if self.current_subgoal_idx < len(self.subgoals) else "")
        next_sg = (self.subgoals[self.current_subgoal_idx + 1]
                    if self.current_subgoal_idx + 1 < len(self.subgoals) else "(end)")
        cache_key = (self.current_instruction_idx, self.current_subgoal_idx)
        if cache_key in self.imagination_cache:
            return self.imagination_cache[cache_key]
        prompt = _load_prompt("imagination.txt").format(
            current_subgoal=subgoal, landmarks=landmarks, next_subgoal=next_sg
        )
        response = self.llm.query_text(prompt)
        parsed = self.llm.extract_json(response)
        if parsed and isinstance(parsed, dict):
            state = parsed.get("state", "")
            self.imagination_cache[cache_key] = state
            return state
        self.imagination_cache[cache_key] = response or ""
        return response or ""

    # ─── Module 6: Subgoal Judger ───

    def subgoal_judger(self, perception_output):
        subgoal = (self.subgoals[self.current_subgoal_idx]
                    if self.current_subgoal_idx < len(self.subgoals) else "reach target")
        next_sg = (self.subgoals[self.current_subgoal_idx + 1]
                    if self.current_subgoal_idx + 1 < len(self.subgoals) else "(end)")
        imagination_state = self.imagination()
        prompt = _load_prompt("subgoal_judger.txt").format(
            current_subgoal=subgoal,
            subgoal_memory=self.subgoal_memory or "(none yet)",
            perception_output=perception_output if perception_output else "(no observation)",
            imagination_state=imagination_state if imagination_state else "(no imagination reference)",
            next_subgoal=next_sg
        )
        response = self.llm.query_text(prompt)
        parsed = self.llm.extract_json(response)
        if parsed and isinstance(parsed, dict):
            return parsed.get("achieved", False), parsed.get("reason", "")
        return False, response or ""

    # ─── Module 7: Decision-Making ───

    def decision_making(self, perception_output, depth_img):
        warnings = self.collision.check(depth_img)
        warn_str = self.collision.format_warning(warnings)
        subgoal = (self.subgoals[self.current_subgoal_idx]
                    if self.current_subgoal_idx < len(self.subgoals) else "reach target")
        cur_inst = (self.parsed_instructions[self.current_instruction_idx]["sub_instruction"]
                     if self.current_instruction_idx < len(self.parsed_instructions) else "navigate")
        prompt = _load_prompt("decision_making.txt").format(
            instruction_memory=self.instruction_memory or "(none yet)",
            perception_output=perception_output if perception_output else "(no observation)",
            collision_warning=warn_str,
            current_instruction=cur_inst,
            current_subgoal=subgoal
        )
        response = self.llm.query_text(prompt)
        parsed = self.llm.extract_json(response)
        if parsed and isinstance(parsed, dict) and "selected_action" in parsed:
            action = int(parsed["selected_action"])
            thought = parsed.get("thought", "")
            return action, thought
        resp_lower = response.lower() if response else ""
        if "stop" in resp_lower or "finish" in resp_lower:
            return 0, response or ""
        if "forward" in resp_lower:
            return 1, response or ""
        if "turn left" in resp_lower or resp_lower.startswith("2"):
            return 2, response or ""
        if "turn right" in resp_lower or resp_lower.startswith("3"):
            return 3, response or ""
        if "ascend" in resp_lower or "go up" in resp_lower or resp_lower.startswith("4"):
            return 4, response or ""
        if "descend" in resp_lower or "go down" in resp_lower or resp_lower.startswith("5"):
            return 5, response or ""
        if "move left" in resp_lower:
            return 6, response or ""
        if "move right" in resp_lower:
            return 7, response or ""
        return 1, response or ""

    # ─── Module 8: Step Memory + Subgoal Consolidation ───

    def step_memory(self, perception_output, action_code):
        cur = self.parsed_instructions[self.current_instruction_idx] if self.current_instruction_idx < len(self.parsed_instructions) else None
        landmarks = ", ".join(cur.get("landmarks", [])) if cur else ""
        action_names = {0: "STOP", 1: "MOVE_FORWARD", 2: "TURN_LEFT", 3: "TURN_RIGHT",
                        4: "ASCEND", 5: "DESCEND", 6: "MOVE_LEFT", 7: "MOVE_RIGHT"}
        action_name = action_names.get(action_code, "UNKNOWN")
        prompt = _load_prompt("memory.txt").split("--- SUBGOAL MEMORY CONSOLIDATION")[0].format(
            perception_output=perception_output if perception_output else "(no observation)",
            action_name=action_name,
            landmarks=landmarks
        )
        response = self.llm.query_text(prompt)
        parsed = self.llm.extract_json(response)
        if parsed and isinstance(parsed, dict):
            sm = parsed.get("step_memory", response or "")
        else:
            sm = response or f"I see ...; I {action_name}"
        self.step_memories.append(sm)
        return sm

    def consolidate_subgoal_memory(self):
        if not self.step_memories:
            return ""
        raw_text = "\n".join(f"{i+1}. {m}" for i, m in enumerate(self.step_memories))
        prompt = f"""[TASK]: Consolidate the raw step memories below into a concise subgoal summary.

[RAW STEP MEMORIES]:
{raw_text}

[RULES]:
1. Merge consecutive SAME actions (e.g., "move forward" x4 → "move forward 20m")
2. "turn left" and "turn right" are DIFFERENT - do NOT merge across each other
3. Preserve temporal order
4. Only include VISIBLE landmarks from the memories
5. Note landmark changes across steps

[OUTPUT]: JSON: {{"subgoal_memory": "I have seen ...; I ...(action summary)"}}
[WORD LIMIT]: Less than 40 words."""
        response = self.llm.query_text(prompt)
        parsed = self.llm.extract_json(response)
        if parsed and isinstance(parsed, dict):
            self.subgoal_memory = parsed.get("subgoal_memory", response or "")
        else:
            self.subgoal_memory = response or "; ".join(self.step_memories[-5:])
        self.instruction_memory += f" [Subgoal done: {self.subgoal_memory}]"
        self.step_memories = []
        return self.subgoal_memory

    # ─── State machine helpers ───

    def advance_subgoal(self):
        """Move to next subgoal; return True if more subgoals remain."""
        self.consolidate_subgoal_memory()
        self.current_subgoal_idx += 1
        if self.current_subgoal_idx >= len(self.subgoals):
            self.current_instruction_idx += 1
            if self.current_instruction_idx >= len(self.parsed_instructions):
                return False
            self.subgoals = []
            self.current_subgoal_idx = 0
            self.imagination_cache.clear()
            return True
        self.imagination_cache.pop(
            (self.current_instruction_idx, self.current_subgoal_idx - 1), None
        )
        return True

    def current_subgoal_text(self):
        if self.current_subgoal_idx < len(self.subgoals):
            return self.subgoals[self.current_subgoal_idx]
        return "reach target"

    def has_more_subgoals(self):
        return (self.current_instruction_idx < len(self.parsed_instructions) and
                self.current_subgoal_idx < len(self.subgoals))
