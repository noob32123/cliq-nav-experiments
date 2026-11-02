"""
download_models.py — Download OpenVLA-7b and AerialVLA LoRA.
Uses HuggingFace mirror (hf-mirror.com) for China mainland users.
Set HF_ENDPOINT to use official HF if you have VPN.
"""
import os
import sys

DOWNLOAD_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models")
BASE_MODEL = "openvla/openvla-7b"
LORA_MODEL = "XuPeng23/AerialVLA"

os.makedirs(os.path.join(DOWNLOAD_DIR, "openvla-7b"), exist_ok=True)
os.makedirs(os.path.join(DOWNLOAD_DIR, "aerialvla_lora"), exist_ok=True)

# Use mirror for China mainland (set HF_ENDPOINT to use official HF)
if "HF_ENDPOINT" not in os.environ:
    os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
    print("Using mirror: hf-mirror.com")
    print("(Set HF_ENDPOINT=https://huggingface.co if you have VPN)\n")


def download_from_hf(repo_id, local_dir):
    """Download from HuggingFace hub."""
    try:
        from huggingface_hub import snapshot_download
        print(f"Downloading {repo_id} → {local_dir} ...")
        print(f"  Endpoint: {os.environ.get('HF_ENDPOINT', 'default')}")
        snapshot_download(
            repo_id=repo_id,
            local_dir=local_dir,
            local_dir_use_symlinks=False,
            resume_download=True,
            ignore_patterns=["*.msgpack", "*.h5", "*.ot", "*.tflite"],
            max_workers=4,
        )
        print(f"Done: {repo_id}\n")
    except ImportError:
        print("huggingface_hub not installed. Run: pip install huggingface_hub")
        sys.exit(1)
    except Exception as e:
        print(f"Download error: {e}")
        print(f"\nManual download steps:")
        print(f"1. Open https://hf-mirror.com/{repo_id}")
        print(f"2. Download all files to:\n   {local_dir}")
        print(f"\nOr use official HuggingFace with VPN:")
        print(f"   set HF_ENDPOINT=https://huggingface.co")
        print(f"   python download_models.py")
        raise


if __name__ == "__main__":
    print("=" * 60)
    print("AerialVLA Model Downloader")
    print("=" * 60)
    print(f"Base model:  {BASE_MODEL} (~14GB)")
    print(f"LoRA adapter: {LORA_MODEL} (~200MB)")
    print(f"Target: {DOWNLOAD_DIR}")
    print()

    # Download base model
    base_dir = os.path.join(DOWNLOAD_DIR, "openvla-7b")
    if os.path.exists(os.path.join(base_dir, "config.json")) and \
       os.path.exists(os.path.join(base_dir, "preprocessor_config.json")):
        print(f"Base model already exists at {base_dir}")
    else:
        download_from_hf(BASE_MODEL, base_dir)

    # Download LoRA adapter
    lora_dir = os.path.join(DOWNLOAD_DIR, "aerialvla_lora")
    if os.path.exists(os.path.join(lora_dir, "adapter_config.json")):
        print(f"LoRA adapter already exists at {lora_dir}")
    else:
        download_from_hf(LORA_MODEL, lora_dir)

    print("\nDone! Models ready at:", DOWNLOAD_DIR)
