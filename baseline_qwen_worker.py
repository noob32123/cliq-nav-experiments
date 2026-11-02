"""One bounded Qwen request. Credentials stay in the inherited environment."""
import json
import os
import sys
from openai import OpenAI


def main():
    request = json.load(sys.stdin)
    model = request.get('model', 'qwen3.5-35b-a3b')
    try:
        with OpenAI(api_key=os.environ['OPENAI_API_KEY'],
                    base_url='https://api.desire-drive.com/v1', max_retries=0, timeout=150) as client:
            response = client.chat.completions.create(
                model=request.get('model', 'qwen3.5-35b-a3b'), messages=request['messages'],
                max_tokens=request['max_tokens'], timeout=150,
                **({'extra_body': {'chat_template_kwargs': {'enable_thinking': False}}}
                   if model.lower().startswith('qwen') else {}))
            if model == 'internvl2-26b' and response.model != model:
                print(json.dumps({'ok': False, 'error_type': 'ResponseModelMismatch',
                                  'http_status': 422, 'response_model': response.model}))
                return
            print(json.dumps({'ok': True, 'text': response.choices[0].message.content or '',
                              'requested_model': request.get('model', 'qwen3.5-35b-a3b'),
                              'response_model': response.model,
                              'base_url': 'https://api.desire-drive.com/v1',
                              'usage': response.usage.model_dump() if response.usage else None}))
    except Exception as exc:
        print(json.dumps({'ok': False, 'error_type': type(exc).__name__,
                          'error_code': ('image_limit_one' if 'At most 1 image(s)' in str(exc) else None),
                          'http_status': getattr(exc, 'status_code', None)}))


if __name__ == '__main__':
    main()
