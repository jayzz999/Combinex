"""
Model download script

Downloads all required models for the pipeline
"""

import os
from pathlib import Path
from huggingface_hub import hf_hub_download, snapshot_download
import argparse


def download_insightface_models(models_dir: Path):
    """Download InsightFace antelopev2 models."""
    print("\n📥 Downloading InsightFace models...")

    insightface_dir = models_dir / "insightface" / "models" / "antelopev2"
    insightface_dir.mkdir(parents=True, exist_ok=True)

    model_files = [
        "1k3d68.onnx",
        "2d106det.onnx",
        "genderage.onnx",
        "glintr100.onnx",
        "scrfd_10g_bnkps.onnx",
    ]

    base_url = "https://huggingface.co/datasets/Gourieff/ReActor/resolve/main/models/insightface/models/antelopev2"

    for model_file in model_files:
        output_path = insightface_dir / model_file

        if output_path.exists():
            print(f"  ✓ {model_file} (already exists)")
            continue

        print(f"  Downloading {model_file}...")
        os.system(f'wget -q -O {output_path} {base_url}/{model_file}')
        print(f"  ✓ {model_file}")

    print("✅ InsightFace models downloaded")


def download_ip_adapter_models(models_dir: Path):
    """Download IP-Adapter-FaceID models."""
    print("\n📥 Downloading IP-Adapter-FaceID models...")

    ip_adapter_dir = models_dir / "ip-adapter-faceid"
    ip_adapter_dir.mkdir(parents=True, exist_ok=True)

    models = [
        "ip-adapter-faceid_sdxl.bin",
        "ip-adapter-faceid-plusv2_sdxl.bin",
    ]

    for model_file in models:
        output_path = ip_adapter_dir / model_file

        if output_path.exists():
            print(f"  ✓ {model_file} (already exists)")
            continue

        print(f"  Downloading {model_file}...")
        hf_hub_download(
            repo_id="h94/IP-Adapter-FaceID",
            filename=model_file,
            local_dir=str(ip_adapter_dir),
        )
        print(f"  ✓ {model_file}")

    print("✅ IP-Adapter-FaceID models downloaded")


def download_base_models(cache_only: bool = True):
    """
    Download/cache base models.

    Args:
        cache_only: If True, just cache to HF cache (don't copy to models dir)
    """
    print("\n📥 Downloading base models to Hugging Face cache...")

    models_to_cache = [
        "stabilityai/stable-diffusion-xl-base-1.0",
        "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
    ]

    for model_id in models_to_cache:
        print(f"  Caching {model_id}...")
        try:
            snapshot_download(
                repo_id=model_id,
                cache_dir=None,  # Use default HF cache
                local_files_only=False,
            )
            print(f"  ✓ {model_id}")
        except Exception as e:
            print(f"  ⚠️ {model_id}: {e}")

    print("✅ Base models cached")


def create_models_readme(models_dir: Path):
    """Create README in models directory."""

    readme_content = """# Models Directory

This directory contains the downloaded models for the Inflight Selfie Generator.

## Structure

```
models/
├── insightface/
│   └── models/
│       └── antelopev2/          # Face analysis models
│           ├── 1k3d68.onnx
│           ├── 2d106det.onnx
│           ├── genderage.onnx
│           ├── glintr100.onnx
│           └── scrfd_10g_bnkps.onnx
├── ip-adapter-faceid/           # IP-Adapter weights
│   ├── ip-adapter-faceid_sdxl.bin
│   └── ip-adapter-faceid-plusv2_sdxl.bin
└── scene_planner_lora/          # Fine-tuned TinyLlama (from Colab)
    ├── adapter_config.json
    └── adapter_model.safetensors
```

## Base Models (Cached)

The following models are downloaded to the Hugging Face cache (~/.cache/huggingface):

- `stabilityai/stable-diffusion-xl-base-1.0` (~7GB)
- `TinyLlama/TinyLlama-1.1B-Chat-v1.0` (~2GB)

These are loaded automatically by the pipeline.

## Fine-tuned Models

After training in Google Colab:

1. Download `inflight_selfie_models.zip` from Colab
2. Extract to this directory:
   ```bash
   unzip inflight_selfie_models.zip -d models/
   ```

## Manual Download

If automatic download fails:

### InsightFace Models
```bash
mkdir -p models/insightface/models/antelopev2
cd models/insightface/models/antelopev2
wget https://huggingface.co/datasets/Gourieff/ReActor/resolve/main/models/insightface/models/antelopev2/1k3d68.onnx
wget https://huggingface.co/datasets/Gourieff/ReActor/resolve/main/models/insightface/models/antelopev2/2d106det.onnx
wget https://huggingface.co/datasets/Gourieff/ReActor/resolve/main/models/insightface/models/antelopev2/genderage.onnx
wget https://huggingface.co/datasets/Gourieff/ReActor/resolve/main/models/insightface/models/antelopev2/glintr100.onnx
wget https://huggingface.co/datasets/Gourieff/ReActor/resolve/main/models/insightface/models/antelopev2/scrfd_10g_bnkps.onnx
```

### IP-Adapter-FaceID
```bash
huggingface-cli download h94/IP-Adapter-FaceID --local-dir models/ip-adapter-faceid
```

## Storage Requirements

- InsightFace: ~300MB
- IP-Adapter-FaceID: ~1GB
- SDXL Base (cached): ~7GB
- TinyLlama (cached): ~2GB
- Fine-tuned LoRA: ~50MB

**Total**: ~10-11GB
"""

    readme_path = models_dir / "README.md"
    with open(readme_path, "w") as f:
        f.write(readme_content)

    print(f"✅ Created models README: {readme_path}")


def main():
    parser = argparse.ArgumentParser(description="Download models for Inflight Selfie Generator")

    parser.add_argument(
        "--models-dir",
        type=str,
        default="./models",
        help="Directory to store models"
    )
    parser.add_argument(
        "--skip-insightface",
        action="store_true",
        help="Skip InsightFace download"
    )
    parser.add_argument(
        "--skip-ip-adapter",
        action="store_true",
        help="Skip IP-Adapter download"
    )
    parser.add_argument(
        "--skip-base",
        action="store_true",
        help="Skip base models caching"
    )

    args = parser.parse_args()

    models_dir = Path(args.models_dir)
    models_dir.mkdir(parents=True, exist_ok=True)

    print("📦 Downloading Models for Inflight Selfie Generator")
    print("=" * 60)
    print(f"Models directory: {models_dir.absolute()}")

    # Download models
    if not args.skip_insightface:
        download_insightface_models(models_dir)

    if not args.skip_ip_adapter:
        download_ip_adapter_models(models_dir)

    if not args.skip_base:
        download_base_models()

    # Create README
    create_models_readme(models_dir)

    print("\n" + "=" * 60)
    print("✅ Model download complete!")
    print("\nNext steps:")
    print("1. Train in Google Colab (notebooks/Inflight_Selfie_Training.ipynb)")
    print("2. Download and extract scene_planner_lora to models/")
    print("3. Run the server: uvicorn server:app --host 0.0.0.0 --port 8000")
    print("=" * 60)


if __name__ == "__main__":
    main()
