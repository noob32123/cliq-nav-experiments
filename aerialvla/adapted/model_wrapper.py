"""
AerialVLA Model Wrapper — loads OpenVLA-7b + LoRA adapter, runs INT8 inference.

Key details:
- Base: openvla/openvla-7b (Prismatic VLM: SigLIP vision + LLaMA)
- Adapter: XuPeng23/AerialVLA (LoRA fine-tuned on TravelUAV)
- Input: vertically stacked 224×448 mosaic (front 224×224 + down 224×224)
- Output: 3-DoF continuous action (fwd∈[0,5]m/s, down∈[-5,5]m/s, yaw∈[-1.1,1.1]rad/s)
- Quantization: INT8 via bitsandbytes (for 16GB GPU)
"""
import os
import re
import numpy as np
from PIL import Image


class AerialVLAWrapper:
    NUM_BINS = 99
    NORM_STATS = {
        "forward": {"min": 0.0, "max": 5.0},
        "down": {"min": -5.0, "max": 5.0},
        "yaw": {"min": -1.1, "max": 1.1},
    }

    def __init__(self, base_model_path, adapter_path, device="cuda", load_in_8bit=True):
        self.device = device
        self.base_path = base_model_path
        self.adapter_path = adapter_path
        print(f"[AerialVLA] Loading model...")

        import torch
        from transformers import AutoTokenizer, AutoImageProcessor, AutoConfig, PreTrainedModel
        from transformers.dynamic_module_utils import get_class_from_dynamic_module
        from peft import PeftModel

        base_local = os.path.abspath(base_model_path)
        adapter_local = os.path.abspath(adapter_path)

        config = AutoConfig.from_pretrained(base_local, trust_remote_code=True, local_files_only=True)
        # Resolve custom model class via auto_map
        class_ref = config.auto_map.get("AutoModelForVision2Seq",
                    config.auto_map.get("AutoModel", None))
        if class_ref:
            model_cls = get_class_from_dynamic_module(class_ref, base_local)
        else:
            from transformers import AutoModel
            model_cls = AutoModel

        self.tokenizer = AutoTokenizer.from_pretrained(base_local, trust_remote_code=True, local_files_only=True)
        self.image_processor = AutoImageProcessor.from_pretrained(base_local, trust_remote_code=True, local_files_only=True)

        model_kwargs = {
            "dtype": torch.bfloat16,
            "low_cpu_mem_usage": True,
            "trust_remote_code": True,
            "attn_implementation": "eager",
        }

        if load_in_8bit:
            from transformers import BitsAndBytesConfig
            model_kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_8bit=True,
                llm_int8_skip_modules=["vision_tower"],
            )
            print("[AerialVLA] Using INT8 quantization")

        self.model = model_cls.from_pretrained(base_local, **model_kwargs)
        self.model.resize_token_embeddings(len(self.tokenizer))
        self.model = PeftModel.from_pretrained(self.model, adapter_local)
        self.model.to(device)
        self.model.eval()

        self.model_dtype = next(self.model.parameters()).dtype
        print(f"[AerialVLA] Model loaded. dtype={self.model_dtype}")

    def _build_mosaic(self, front_rgb, down_rgb):
        """Resize both views to 224×224 and vertically stack into 224×448."""
        target = (224, 224)
        if isinstance(front_rgb, np.ndarray):
            front = Image.fromarray(front_rgb)
        else:
            front = front_rgb
        if isinstance(down_rgb, np.ndarray):
            down = Image.fromarray(down_rgb)
        else:
            down = down_rgb
        front = front.resize(target, resample=Image.BICUBIC)
        down = down.resize(target, resample=Image.BICUBIC)
        mosaic = Image.new('RGB', (224, 448))
        mosaic.paste(front, (0, 0))
        mosaic.paste(down, (0, 224))
        return mosaic

    def _dequantize(self, bin_val, axis):
        stats = self.NORM_STATS[axis]
        v = max(0, min(self.NUM_BINS - 1, bin_val)) / (self.NUM_BINS - 1)
        return v * (stats["max"] - stats["min"]) + stats["min"]

    def run(self, front_rgb, down_rgb, prompt_text):
        import torch

        mosaic = self._build_mosaic(front_rgb, down_rgb)
        pv = self.image_processor(images=mosaic, return_tensors='pt')['pixel_values']
        if hasattr(self, 'model_dtype'):
            pv = pv.to(device=self.device, dtype=self.model_dtype)
        else:
            pv = pv.to(device=self.device)

        inputs = self.tokenizer(prompt_text, return_tensors="pt").to(self.device)
        inputs['pixel_values'] = pv

        with torch.no_grad():
            outputs = self.model.generate(**inputs, max_new_tokens=20, do_sample=False)

        text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)

        action_part = text.split("Action:")[-1] if "Action:" in text else text
        numbers = re.findall(r"\d+", action_part)
        if len(numbers) >= 3:
            bins = list(map(int, numbers[-3:]))
        else:
            bins = [49, 49, 49]

        fwd = self._dequantize(bins[0], "forward")
        down = self._dequantize(bins[1], "down")
        yaw = self._dequantize(bins[2], "yaw")

        has_land_str = "LAND" in text.upper() or "<LAND>" in text
        has_land_num = (abs(fwd) < 0.01 and abs(down) < 0.01 and abs(yaw) < 0.01)
        has_land = has_land_str or has_land_num

        return {
            "text": text,
            "fwd": fwd,
            "down": down,
            "yaw": yaw,
            "bins": bins,
            "land": has_land,
        }
