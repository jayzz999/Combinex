"""
Dataset preparation script for IP-Adapter-FaceID fine-tuning

Creates the dataset structure needed for training:
- Face reference images
- Target inflight selfie images
- Metadata JSON file
"""

import os
import json
import shutil
from pathlib import Path
from typing import List, Dict
import argparse


# Sample training data prompts
INFLIGHT_SCENE_PROMPTS = [
    {
        "scene_type": "sunset_window",
        "prompt": "two friends taking selfie in airplane window seat, golden sunset visible through window, Dubai skyline approaching in distance, warm golden lighting, happy excited expressions, economy cabin, airplane interior, high quality photo, realistic, detailed faces",
        "negative_prompt": "ugly, blurry, distorted, bad anatomy, deformed faces",
        "ip_adapter_scale": 0.65,
        "guidance_scale": 7.5,
    },
    {
        "scene_type": "business_celebration",
        "prompt": "two people taking selfie in business class airplane cabin, champagne glasses in hand, celebrating, luxury wide seats, premium cabin interior, soft ambient lighting, joyful expressions, high quality photo, detailed, realistic",
        "negative_prompt": "ugly, blurry, economy class, cheap, low quality",
        "ip_adapter_scale": 0.6,
        "guidance_scale": 7.0,
    },
    {
        "scene_type": "night_city",
        "prompt": "two people taking selfie in airplane at night, city lights visible through window below, dim cabin lighting with city glow on faces, amazed expressions, window seat, high quality photo, realistic, atmospheric",
        "negative_prompt": "daylight, bright, ugly, blurry, bad lighting",
        "ip_adapter_scale": 0.7,
        "guidance_scale": 7.5,
    },
    {
        "scene_type": "morning_clouds",
        "prompt": "two people taking selfie in airplane window seat, fluffy white clouds visible outside, bright morning sunlight, fresh morning mood, happy smiling faces, economy cabin, high quality photo, realistic",
        "negative_prompt": "dark, night, ugly, blurry, bad quality",
        "ip_adapter_scale": 0.65,
        "guidance_scale": 7.0,
    },
    {
        "scene_type": "first_class",
        "prompt": "two people taking selfie in first class airplane suite, spacious luxury cabin, premium amenities visible, elegant lighting, sophisticated expressions, high-end travel experience, high quality photo, detailed, realistic",
        "negative_prompt": "cheap, low quality, economy, cramped, ugly",
        "ip_adapter_scale": 0.6,
        "guidance_scale": 7.5,
    },
    {
        "scene_type": "takeoff",
        "prompt": "two friends taking selfie during airplane takeoff, runway visible through window, excited nervous expressions, beginning of journey mood, window seat, high quality photo, realistic, dynamic moment",
        "negative_prompt": "calm, boring, ugly, blurry, static",
        "ip_adapter_scale": 0.7,
        "guidance_scale": 7.5,
    },
    {
        "scene_type": "red_eye",
        "prompt": "two travelers taking selfie during red eye flight, tired but happy expressions, blankets visible, dim cabin lighting, night atmosphere, cozy travel mood, high quality photo, realistic",
        "negative_prompt": "energetic, bright, ugly, blurry",
        "ip_adapter_scale": 0.65,
        "guidance_scale": 7.0,
    },
    {
        "scene_type": "landing",
        "prompt": "two people taking selfie during airplane landing, destination airport visible through window, excited arrival expressions, end of journey celebration, window seat, high quality photo, realistic, arrival mood",
        "negative_prompt": "departure, ugly, blurry, bad quality",
        "ip_adapter_scale": 0.65,
        "guidance_scale": 7.5,
    },
    {
        "scene_type": "tropical",
        "prompt": "two friends taking selfie in airplane, tropical ocean and islands visible through window, bright blue water below, vacation excitement mood, happy expressions, window seat, high quality photo, realistic, travel adventure",
        "negative_prompt": "ugly, blurry, dark, winter, mountains",
        "ip_adapter_scale": 0.65,
        "guidance_scale": 7.0,
    },
    {
        "scene_type": "business",
        "prompt": "two business colleagues taking selfie in airplane, professional friendly expressions, business casual attire, business travel atmosphere, modern cabin, high quality photo, realistic, professional mood",
        "negative_prompt": "casual, party, ugly, blurry, unprofessional",
        "ip_adapter_scale": 0.6,
        "guidance_scale": 7.5,
    },
]


def create_dataset_structure(output_dir: str = "./dataset") -> Path:
    """
    Create the dataset directory structure.

    Args:
        output_dir: Output directory path

    Returns:
        Path to dataset directory
    """
    dataset_path = Path(output_dir)

    # Create directories
    (dataset_path / "images").mkdir(parents=True, exist_ok=True)
    (dataset_path / "face_refs").mkdir(parents=True, exist_ok=True)

    print(f"✅ Created dataset structure at: {dataset_path}")
    print(f"   - {dataset_path / 'images'}/     (target selfie images)")
    print(f"   - {dataset_path / 'face_refs'}/  (face reference images)")

    return dataset_path


def generate_metadata(
    dataset_path: Path,
    num_samples: int = None,
) -> Dict:
    """
    Generate metadata.json for training.

    Args:
        dataset_path: Path to dataset directory
        num_samples: Number of samples to include (None = all)

    Returns:
        Metadata dictionary
    """
    prompts = INFLIGHT_SCENE_PROMPTS[:num_samples] if num_samples else INFLIGHT_SCENE_PROMPTS

    metadata = []
    for i, prompt_data in enumerate(prompts):
        metadata.append({
            "id": i,
            "scene_type": prompt_data["scene_type"],
            "prompt": prompt_data["prompt"],
            "negative_prompt": prompt_data["negative_prompt"],
            "ip_adapter_scale": prompt_data["ip_adapter_scale"],
            "guidance_scale": prompt_data["guidance_scale"],
            "face_image": f"face_refs/face_{i:03d}.jpg",
            "target_image": f"images/target_{i:03d}.jpg",
        })

    # Save metadata
    metadata_path = dataset_path / "metadata.json"
    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=2)

    print(f"✅ Generated metadata.json with {len(metadata)} samples")
    print(f"   Saved to: {metadata_path}")

    return metadata


def create_readme(dataset_path: Path):
    """Create README for dataset directory."""

    readme_content = """# Inflight Selfie Dataset

## Structure

```
dataset/
├── images/          # Target inflight selfie images
│   ├── target_000.jpg
│   ├── target_001.jpg
│   └── ...
├── face_refs/       # Face reference images
│   ├── face_000.jpg
│   ├── face_001.jpg
│   └── ...
├── metadata.json    # Training metadata
└── README.md        # This file
```

## Adding Your Data

### Option 1: Manual Collection

1. **Collect real inflight selfies** or **generate synthetic data**
   - Search online for inflight selfie images
   - Use existing SDXL to generate base images
   - Ensure images show airplane cabin interior

2. **Add to dataset**:
   ```bash
   # Add target images
   cp your_selfie1.jpg dataset/images/target_000.jpg
   cp your_selfie2.jpg dataset/images/target_001.jpg

   # Add corresponding face references
   cp face_ref1.jpg dataset/face_refs/face_000.jpg
   cp face_ref2.jpg dataset/face_refs/face_001.jpg
   ```

3. **Update metadata.json** if needed

### Option 2: Synthetic Generation

Use SDXL to generate initial dataset:

```python
from diffusers import StableDiffusionXLPipeline
import torch

pipe = StableDiffusionXLPipeline.from_pretrained(
    "stabilityai/stable-diffusion-xl-base-1.0",
    torch_dtype=torch.float16
).to("cuda")

# Generate for each prompt in metadata
for item in metadata:
    image = pipe(prompt=item["prompt"]).images[0]
    image.save(f"dataset/{item['target_image']}")
```

### Option 3: Use Existing Datasets

- Search Hugging Face datasets for airplane/travel images
- Filter for cabin interior selfies
- Augment with face swapping if needed

## Metadata Format

```json
[
  {
    "id": 0,
    "scene_type": "sunset_window",
    "prompt": "detailed scene description...",
    "negative_prompt": "what to avoid...",
    "ip_adapter_scale": 0.65,
    "guidance_scale": 7.5,
    "face_image": "face_refs/face_000.jpg",
    "target_image": "images/target_000.jpg"
  }
]
```

## Best Practices

1. **Image Quality**: Use high-resolution images (1024x1024 minimum)
2. **Consistency**: Ensure face references match target images
3. **Diversity**: Include various scenes (sunset, night, business class, etc.)
4. **Quantity**: 10-50 samples recommended for fine-tuning

## Training

Once dataset is ready, use the Colab notebook for fine-tuning:

1. Upload dataset to Google Drive
2. Mount in Colab
3. Point training script to this directory
4. Run fine-tuning cells

## Notes

- This is a **placeholder dataset** - you need to add real images
- For production quality, collect 50-100+ diverse samples
- Consider data augmentation (crops, flips, color adjustments)
"""

    readme_path = dataset_path / "README.md"
    with open(readme_path, "w") as f:
        f.write(readme_content)

    print(f"✅ Created dataset README: {readme_path}")


def main():
    parser = argparse.ArgumentParser(description="Prepare inflight selfie training dataset")
    parser.add_argument(
        "--output-dir",
        type=str,
        default="./dataset",
        help="Output directory for dataset"
    )
    parser.add_argument(
        "--num-samples",
        type=int,
        default=None,
        help="Number of samples to generate metadata for (default: all)"
    )

    args = parser.parse_args()

    print("🎬 Preparing Inflight Selfie Dataset")
    print("=" * 60)

    # Create structure
    dataset_path = create_dataset_structure(args.output_dir)

    # Generate metadata
    metadata = generate_metadata(dataset_path, args.num_samples)

    # Create README
    create_readme(dataset_path)

    print("\n" + "=" * 60)
    print("✅ Dataset preparation complete!")
    print("\nNext steps:")
    print("1. Add your training images to dataset/images/")
    print("2. Add face reference images to dataset/face_refs/")
    print("3. Review metadata.json")
    print("4. Upload to Google Drive for Colab training")
    print("\nSee dataset/README.md for detailed instructions")


if __name__ == "__main__":
    main()
