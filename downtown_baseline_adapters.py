"""Project-adapted policies; no goal coordinates enter any policy interface."""
import json
import math
import re
from pathlib import Path
import cv2
import numpy as np
import setup_path
import airsim
from city_nav.adapted.citynav_pipeline import CityNavPipeline
from finecog_nav.adapted.finecog_modules import FineCogModules, _load_prompt
from finecog_nav.adapted.llm_client import FineCogLLM
from finecog_nav.adapted.collision_checker import CollisionChecker
from aerialvla.adapted.model_wrapper import AerialVLAWrapper
from aerialvla.adapted.flight_controller import AerialVLAFlightController
from aerialvla.adapted.directional_prompt import build_prompt

ROOT = Path(__file__).resolve().parent
ENGLISH_TARGETS = [
 'the entrance beside the blue and white striped awning on the ground floor of the beige corner building ahead',
 'the recessed entrance below the black BRUN & PEREZ sign ahead, with orange sale posters on both sides',
 'the restaurant entrance ahead beside blue and white striped awnings, dark outdoor tables and chairs, and green columns',
 'the shop entrance ahead with a white door frame and a red Weefit sign',
 'the recessed shop entrance ahead with a blue-gray frame and dark glass',
 'the shop entrance ahead with a red door frame beneath the black EUROPA sign',
 "the entrance ahead between blue and white striped awnings beneath the Jacob's Downtown Deli sign",
 'the doorway beneath a pale blue awning on the ground floor of the light-colored building at the end of the street ahead',
 'the shop entrance beside a yellow SHOP graphic sign on the ground floor of the light-colored building ahead',
 'the shop entrance ahead with a white Sleevo sign and a dark glass storefront',
 'the corner storefront entrance beneath the green awning ahead',
 'the entrance ahead beneath a green awning, with a red tomato logo and the word uFoods',
 'the small doorway ahead covered by an orange awning beneath the UPSWAY sign',
 'the shop entrance ahead beneath the reddish brown awning',
 'the restaurant entrance ahead with red door and window frames, a long green awning, and outdoor tables and chairs',
 'the entrance ahead with a white cafe sign, small red and white striped awning, and yellow door frame',
 'the cafe entrance ahead beneath blue and white striped awnings and a CAFE VENDIE sign with a white coffee cup',
 'the recessed THRIFTY shop entrance ahead',
 'the white-framed recessed entrance ahead beneath the white LOCAFE sign and round coffee-cup logo',
 'the doorway beneath a pale green awning on the ground floor of the light-colored building across the road',
 'the eagle statue in the flower bed ahead; land on open ground outside the flower bed',
 'the deer statue in the flower bed ahead; land on open ground outside the flower bed',
 'the circular park fountain ahead; land on the open paved ground outside the fountain',
 'the food cart under a red and yellow striped umbrella on the pedestrian street ahead; land on open ground beside the cart',
 'the food cart under a red and yellow striped umbrella on the pedestrian street ahead; land on open ground beside the cart',
 'the purple umbrella seating area beside the beige corner building ahead; land on open ground outside the tables and chairs',
 'the resting area with black umbrellas and white tables and chairs beside the corner building ahead; land on open ground outside the seating',
 'the circular fountain ahead, approaching from one side of the park; land on the open paved ground outside the fountain',
 'the wooden bench between two large flower pots on the traffic island ahead; land on open ground in front of the bench',
 'the resting area with purple umbrellas and white tables and chairs beside the building at the end of the street ahead; land on open ground outside the seating',
]


def heading_direction(initial_yaw, current_yaw):
    """Only onboard yaw and the initial facing direction, radians in NED."""
    angle = math.degrees(math.atan2(math.sin(initial_yaw-current_yaw), math.cos(initial_yaw-current_yaw)))
    if -15 < angle <= 15: return 'straight ahead '
    if 15 < angle <= 60: return 'forward-right '
    if 60 < angle <= 120: return 'to your right '
    if 120 < angle <= 180: return 'to your right rear '
    if -60 < angle <= -15: return 'forward-left '
    if -120 < angle <= -60: return 'to your left '
    return 'to your left rear '


def discrete_velocity(code, yaw):
    if code == 1: return (2*math.cos(yaw), 2*math.sin(yaw), 0, 2.5)
    if code == 6: return (2*math.sin(yaw), -2*math.cos(yaw), 0, 2.5)
    if code == 7: return (-2*math.sin(yaw), 2*math.cos(yaw), 0, 2.5)
    if code == 4: return (0, 0, -1.5, 2/1.5)
    if code == 5: return (0, 0, 1.5, 2/1.5)
    raise ValueError('Not a translation action')


def execute_discrete(client, code):
    yaw = airsim.to_eularian_angles(client.simGetVehiclePose().orientation)[2]
    if code in (2, 3):
        heading = math.degrees(yaw) + (-15 if code == 2 else 15)
        client.rotateToYawAsync(heading, timeout_sec=5, margin=1).join()
    else:
        vx, vy, vz, duration = discrete_velocity(code, yaw)
        client.moveByVelocityAsync(vx, vy, vz, duration,
            drivetrain=airsim.DrivetrainType.MaxDegreeOfFreedom,
            yaw_mode=airsim.YawMode(False, math.degrees(yaw))).join()


class PoseAdapter:
    def __init__(self, client): self.client = client
    def get_pose(self): return self.client.simGetVehiclePose()
    def get_position(self): return np.array(list(self.get_pose().position))
    def get_orientation_quat(self):
        q = self.get_pose().orientation
        return np.array([q.w_val, q.x_val, q.y_val, q.z_val])


class LoggedLLM:
    extract_json = staticmethod(FineCogLLM.extract_json)
    def __init__(self, batch): self.batch = batch
    def query_text(self, prompt, max_tokens=1000):
        return self.batch.request([{'role': 'user', 'content': prompt}], max_tokens)
    def query_vision(self, prompt, image_np, max_tokens=800):
        return self.batch.request([{'role': 'user', 'content': [self.image(image_np), {'type': 'text', 'text': prompt}]}], max_tokens)
    def image(self, arr):
        import nav_qwen_zgc as nav
        return {'type': 'image_url', 'image_url': {'url': nav.encode_numpy_to_data_url(arr)}}


class CityPolicy:
    def __init__(self, batch, target, initial_yaw):
        self.batch, self.target = batch, target
        self.pipeline = CityNavPipeline(None, None, None, None)
    def decide(self, images, yaw):
        prompt = self.pipeline.build_citynav_prompt('Navigate to '+self.target, self.target)
        # Correct the sensor legend and unsupported action, not the planning strategy.
        prompt = prompt.replace('brighter = closer objects', 'brighter = farther objects')
        prompt = prompt.replace('MOVE_FORWARD/MOVE_BACKWARD', 'MOVE_FORWARD')
        llm = LoggedLLM(self.batch)
        content = []
        views = [('front_rgb', 'front'), ('front_depth', 'depth_vis'), ('top_down', 'down')]
        if self.batch.state.get('observation_mode') == 'front_rgb_only':
            prompt = re.sub(r'You have 3 camera views provided above:.*?Your task:',
                            'You have one front-facing RGB camera image provided above.\n\nYour task:',
                            prompt, flags=re.S)
            prompt = prompt.replace('the 3 images above', 'the front-facing RGB image above')
            views = [('front_rgb', 'front')]
        for label, key in views:
            content.extend([{'type': 'text', 'text': label+' image:'}, llm.image(images[key])])
        text = self.batch.request([{'role': 'user', 'content': content+[{'type': 'text', 'text': prompt}]}], 200)
        match = re.search(r'^\s*Action:\s*(\w+)', text, re.I | re.M)
        name = match.group(1).upper() if match else text.strip().upper()
        code = self.pipeline.ACTION_CODES.get(name, -1)
        if code >= 0: self.pipeline.action_history.append(name)
        return {'code': code, 'text': text, 'stop': code == 0}


class StrictFineModules(FineCogModules):
    def decision_making(self, perception_output, depth_img):
        warnings = self.collision.check(depth_img)
        cur = self.parsed_instructions[self.current_instruction_idx]['sub_instruction']
        prompt = _load_prompt('decision_making.txt').format(
            instruction_memory=self.instruction_memory or '(none yet)',
            perception_output=perception_output or '(no observation)',
            collision_warning=self.collision.format_warning(warnings),
            current_instruction=cur, current_subgoal=self.current_subgoal_text())
        text = self.llm.query_text(prompt)
        obj = self.llm.extract_json(text)
        code = obj.get('selected_action') if isinstance(obj, dict) else None
        if isinstance(code, str) and code.isdigit(): code = int(code)
        if type(code) is not int or code not in range(8): return -1, text
        return code, text


class FinePolicy:
    def __init__(self, batch, target, initial_yaw):
        self.batch, self.target = batch, target
        self.modules = StrictFineModules(LoggedLLM(batch), None, CollisionChecker(), None)
        self.initialized = False
    def decide(self, images, yaw):
        m = self.modules
        if not self.initialized:
            m.instruction_parser('Navigate to '+self.target)
            self.initialized = True
        perception = m.perception(images['front'])
        if not m.subgoals: m.subgoal_extractor(perception)
        achieved, reason = m.subgoal_judger(perception)
        if achieved is True and m.has_more_subgoals():
            if not m.advance_subgoal(): return {'code': 0, 'stop': True, 'text': reason}
        code, text = m.decision_making(perception, images['depth_m'])
        if code >= 0: m.step_memory(perception, code)
        return {'code': code, 'stop': code == 0, 'text': text}


class LocalAerialModel(AerialVLAWrapper):
    def __init__(self):
        import sys
        runtime = ROOT/'.baseline_runtime/packages'
        if not runtime.exists(): raise RuntimeError('AerialVLA isolated runtime is missing')
        sys.path.insert(0, str(runtime))
        import os
        os.environ['HF_MODULES_CACHE'] = str(ROOT/'.baseline_runtime/hf_modules')
        import torch
        import transformers
        if transformers.__version__ != '4.46.3':
            raise RuntimeError('AerialVLA requires its isolated Transformers 4.46.3 runtime')
        from transformers import AutoTokenizer, AutoImageProcessor, AutoConfig, BitsAndBytesConfig
        from transformers.dynamic_module_utils import get_class_from_dynamic_module
        from peft import PeftModel
        self.device = 'cuda'
        base = str(ROOT/'aerialvla/models/openvla-7b')
        adapter = str(ROOT/'aerialvla/models/aerialvla_lora/aerial_vla')
        cfg = AutoConfig.from_pretrained(base, trust_remote_code=True, local_files_only=True)
        cls = get_class_from_dynamic_module(cfg.auto_map['AutoModelForVision2Seq'], base)
        class QuantizedPrismatic(cls):
            def prepare_inputs_for_generation(self, input_ids=None, past_key_values=None, **kwargs):
                # HF 5 preallocates an empty DynamicCache before the first token.
                # The original HF 4 wrapper treats every non-None cache as a later
                # step, truncating away the prompt and skipping visual perception.
                if past_key_values is not None and hasattr(past_key_values, 'get_seq_length'):
                    if past_key_values.get_seq_length() == 0:
                        past_key_values = None
                return super().prepare_inputs_for_generation(
                    input_ids=input_ids, past_key_values=past_key_values, **kwargs)

            def _init_weights(self, module):
                # Remote Prismatic assumes every Linear weight is floating point.
                # Transformers 5 visits already quantized Linear8bitLt modules too.
                weight = getattr(module, 'weight', None)
                if weight is not None and not weight.is_floating_point():
                    return
                return super()._init_weights(module)
        self.tokenizer = AutoTokenizer.from_pretrained(base, trust_remote_code=True, local_files_only=True)
        self.image_processor = AutoImageProcessor.from_pretrained(base, trust_remote_code=True, local_files_only=True)
        model, info = QuantizedPrismatic.from_pretrained(base, torch_dtype=torch.bfloat16, low_cpu_mem_usage=True,
            attn_implementation='eager', local_files_only=True, device_map={'': 0},
            quantization_config=BitsAndBytesConfig(load_in_8bit=True,
                llm_int8_skip_modules=['projector', 'vision_backbone', 'lm_head']), output_loading_info=True)
        if info.get('missing_keys') or info.get('mismatched_keys') or info.get('error_msgs'):
            raise RuntimeError('Incomplete AerialVLA base checkpoint loading: '+repr(info))
        model.resize_token_embeddings(len(self.tokenizer))
        import bitsandbytes as bnb
        for module in model.modules():
            if isinstance(module, bnb.nn.Linear8bitLt) and not hasattr(module.state, 'memory_efficient_backward'):
                # PEFT 0.13 reads this retired, inference-irrelevant bnb flag.
                module.state.memory_efficient_backward = False
        self.model = PeftModel.from_pretrained(model, adapter, local_files_only=True)
        self.model.eval()
        self.model_dtype = torch.bfloat16
        self.visual_forward_count = 0
        def count_visual(module, args, output): self.visual_forward_count += 1
        model.vision_backbone.register_forward_hook(count_visual)

    def run(self, front_rgb, down_rgb, prompt_text):
        import torch
        mosaic = self._build_mosaic(front_rgb, down_rgb)
        pixels = self.image_processor(images=mosaic, return_tensors='pt')['pixel_values'].to('cuda', dtype=self.model_dtype)
        inputs = self.tokenizer(prompt_text, return_tensors='pt').to('cuda')
        inputs['pixel_values'] = pixels
        before = self.visual_forward_count
        with torch.inference_mode():
            output = self.model.generate(**inputs, max_new_tokens=20, do_sample=False)
        if self.visual_forward_count == before:
            raise RuntimeError('AerialVLA generation skipped image encoding')
        text = self.tokenizer.decode(output[0, inputs['input_ids'].shape[1]:], skip_special_tokens=True)
        return self.decode_action(text)

    def decode_action(self, text):
        if re.fullmatch(r'\s*<?LAND>?[.!]?\s*', text, re.I):
            return {'text': text, 'stop': True, 'valid': True}
        numbers = re.findall(r'(?<![\w.])-?\d+(?![\w.])', text)
        if len(numbers) != 3 or any(not 0 <= int(x) <= 98 for x in numbers):
            return {'text': text, 'stop': False, 'valid': False}
        bins = list(map(int, numbers))
        fwd, down, yaw = [self._dequantize(b, axis) for b, axis in zip(bins, ['forward', 'down', 'yaw'])]
        return {'text': text, 'valid': True, 'bins': bins, 'fwd': fwd, 'down': down, 'yaw': yaw,
                'stop': abs(fwd) < .01 and abs(down) < .01 and abs(yaw) < .01}


class AerialPolicy:
    def __init__(self, batch, target, initial_yaw):
        self.batch, self.target, self.initial_yaw = batch, target, initial_yaw
    def decide(self, images, yaw):
        prompt = build_prompt(heading_direction(self.initial_yaw, yaw), self.target)
        self.batch.event('local_model_input', step=self.batch.step, prompt=prompt)
        return self.batch.local_model.run(cv2.cvtColor(images['front'], cv2.COLOR_BGR2RGB),
                                         cv2.cvtColor(images['down'], cv2.COLOR_BGR2RGB), prompt)


POLICIES = {'aerialvla': AerialPolicy, 'citynav': CityPolicy, 'finecog': FinePolicy}
