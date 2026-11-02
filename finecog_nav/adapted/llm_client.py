"""
FineCog-Nav LLM client adapter 
for ALL modules (reasoning + visual perception, since this model supports images).
"""
import os
import time
import json
import re
import base64
from openai import OpenAI


class FineCogLLM:
    """Single model for all FineCog-Nav modules (LLM reasoning + VLM perception)."""

    def __init__(self, model="qwen3.5-27b", api_key=None, base_url=None):
        self.model = model
        if api_key is None:
            api_key = os.getenv("DASHSCOPE_API_KEY")
            if not api_key:
                raise RuntimeError("DASHSCOPE_API_KEY not set.")
        if base_url is None:
            base_url = "https://dashscope.aliyuncs.com/compatible-mode/v1"
        self.client = OpenAI(api_key=api_key, base_url=base_url)
        print(f"[FineCogLLM] Initialized model={model}")

    def _encode_image_np(self, img_np):
        import cv2
        success, buffer = cv2.imencode('.jpeg', img_np)
        if not success:
            raise RuntimeError("Image encoding failed")
        return base64.b64encode(buffer).decode('utf-8')

    def _call_api(self, messages, max_tokens=1000, timeout=150):
        for attempt in range(3):
            try:
                completion = self.client.chat.completions.create(
                    model=self.model, messages=messages,
                    max_tokens=max_tokens, timeout=timeout
                )
                return completion.choices[0].message.content
            except Exception as e:
                print(f"[FineCogLLM] Attempt {attempt + 1} failed: {e}")
                if attempt < 2:
                    time.sleep(10)
        raise RuntimeError("API call failed after 3 attempts")

    def query_text(self, prompt, max_tokens=1000):
        """Text-only LLM query (used by reasoning modules)."""
        messages = [{"role": "user", "content": prompt}]
        return self._call_api(messages, max_tokens)

    def query_vision(self, prompt, image_np, max_tokens=800):
        """Vision query with one image (used by Perception module)."""
        b64 = self._encode_image_np(image_np)
        content = [
            {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64}"}},
            {"type": "text", "text": prompt}
        ]
        messages = [{"role": "user", "content": content}]
        return self._call_api(messages, max_tokens)

    @staticmethod
    def extract_json(response_text):
        """Extract JSON object from LLM response (handles markdown code blocks)."""
        cleaned = response_text.strip()
        match = re.search(r'```(?:json)?\s*\n?(.*?)\n?```', cleaned, re.DOTALL)
        if match:
            cleaned = match.group(1).strip()
        bracket_start = cleaned.find('{')
        bracket_start_arr = cleaned.find('[')
        if bracket_start_arr >= 0 and (bracket_start < 0 or bracket_start_arr < bracket_start):
            bracket_start = bracket_start_arr
        if bracket_start >= 0:
            cleaned = cleaned[bracket_start:]
            depth = 0
            end = 0
            for i, c in enumerate(cleaned):
                if c in '{[':
                    depth += 1
                elif c in '}]':
                    depth -= 1
                    if depth == 0:
                        end = i + 1
                        break
            if end > 0:
                cleaned = cleaned[:end]
        try:
            return json.loads(cleaned)
        except json.JSONDecodeError:
            try:
                cleaned_single = cleaned.replace("'", '"')
                return json.loads(cleaned_single)
            except json.JSONDecodeError:
                print(f"[FineCogLLM] Failed to parse JSON: {response_text[:300]}")
                return None
