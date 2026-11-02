"""
Qwen LLM client adapter - replaces OpenAI_LLM_v2 (GPT-4o) with Qwen via DashScope.
Uses the same OpenAI-compatible API interface as the user's nav_qwen.py.
"""
import os
import json
import time
import base64
from openai import OpenAI


class QwenLLMAdapter:
    """
    Qwen Vision-Language Model client, API-compatible with OpenAI client.
    Replaces CityNavAgent's OpenAI_LLM_v2 for GPT-4o calls.
    """

    def __init__(self, model="qwen3.5-35b-a3b", api_key=None, base_url=None):
        self.model = model
        if api_key is None:
            api_key = os.getenv("DASHSCOPE_API_KEY")
            if not api_key:
                raise RuntimeError(
                    "DASHSCOPE_API_KEY not set. Set environment variable "
                    "or pass api_key to QwenLLMAdapter()."
                )
        if base_url is None:
            base_url = "https://dashscope.aliyuncs.com/compatible-mode/v1"

        self.client = OpenAI(api_key=api_key, base_url=base_url)
        print(f"[QwenAdapter] Initialized with model={model}")

    def _encode_image(self, image_path):
        """Base64-encode an image file for API."""
        with open(image_path, "rb") as f:
            return base64.b64encode(f.read()).decode("utf-8")

    def _encode_image_np(self, img_np):
        """Base64-encode a numpy image array for API."""
        import cv2
        success, buffer = cv2.imencode('.jpeg', img_np)
        if not success:
            raise RuntimeError("Image encoding failed")
        return base64.b64encode(buffer).decode('utf-8')

    def query_api(self, prompt, image_path=None, max_tokens=1000, timeout=120):
        """
        Text query, optionally with a single image.
        Matching CityNavAgent's query_api() interface.
        """
        content = []

        if image_path is not None:
            if isinstance(image_path, str):
                b64 = self._encode_image(image_path)
            else:
                b64 = self._encode_image_np(image_path)
            content.append({
                "type": "image_url",
                "image_url": {"url": f"data:image/jpeg;base64,{b64}"}
            })

        content.append({"type": "text", "text": prompt})

        messages = [{"role": "user", "content": content}]

        for attempt in range(3):
            try:
                completion = self.client.chat.completions.create(
                    model=self.model,
                    messages=messages,
                    max_tokens=max_tokens,
                    timeout=timeout
                )
                return completion.choices[0].message.content
            except Exception as e:
                print(f"[QwenAdapter] API call attempt {attempt + 1} failed: {e}")
                if attempt < 2:
                    time.sleep(10)

        raise RuntimeError("Qwen API call failed after 3 attempts")

    def query_apis(self, prompt, image_paths=None, max_tokens=1000, timeout=120):
        """
        Text query with multiple images.
        Matching CityNavAgent's query_apis() interface.
        """
        content = []

        if image_paths:
            for img in image_paths:
                if isinstance(img, str):
                    b64 = self._encode_image(img)
                else:
                    b64 = self._encode_image_np(img)
                content.append({
                    "type": "image_url",
                    "image_url": {"url": f"data:image/jpeg;base64,{b64}"}
                })

        content.append({"type": "text", "text": prompt})
        messages = [{"role": "user", "content": content}]

        for attempt in range(3):
            try:
                completion = self.client.chat.completions.create(
                    model=self.model,
                    messages=messages,
                    max_tokens=max_tokens,
                    timeout=timeout
                )
                return completion.choices[0].message.content
            except Exception as e:
                print(f"[QwenAdapter] API call attempt {attempt + 1} failed: {e}")
                if attempt < 2:
                    time.sleep(10)

        raise RuntimeError("Qwen API call failed after 3 attempts")

    def query_viewpoint_api(self, prompt, viewpoint_images, max_tokens=1000, timeout=120):
        """
        Multi-viewpoint query with labeled images.
        Maps CityNavAgent's query_viewpoint_api() interface.
        viewpoint_images: dict of {direction_name: image_np_or_path}
        """
        content = []

        for direction, img in viewpoint_images.items():
            if isinstance(img, str):
                b64 = self._encode_image(img)
            else:
                b64 = self._encode_image_np(img)
            content.append({"type": "text", "text": f"{direction} image:"})
            content.append({
                "type": "image_url",
                "image_url": {"url": f"data:image/jpeg;base64,{b64}"}
            })

        content.append({"type": "text", "text": prompt})
        messages = [{"role": "user", "content": content}]

        for attempt in range(3):
            try:
                completion = self.client.chat.completions.create(
                    model=self.model,
                    messages=messages,
                    max_tokens=max_tokens,
                    timeout=timeout
                )
                return completion.choices[0].message.content
            except Exception as e:
                print(f"[QwenAdapter] API call attempt {attempt + 1} failed: {e}")
                if attempt < 2:
                    time.sleep(10)

        raise RuntimeError("Qwen API call failed after 3 attempts")

    def query_text(self, prompt, max_tokens=1000, timeout=120):
        """Text-only query."""
        messages = [{"role": "user", "content": prompt}]
        for attempt in range(3):
            try:
                completion = self.client.chat.completions.create(
                    model=self.model,
                    messages=messages,
                    max_tokens=max_tokens,
                    timeout=timeout
                )
                return completion.choices[0].message.content
            except Exception as e:
                print(f"[QwenAdapter] Text call attempt {attempt + 1} failed: {e}")
                if attempt < 2:
                    time.sleep(10)
        raise RuntimeError("Qwen text call failed after 3 attempts")
