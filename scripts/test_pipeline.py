"""
Test script for Inflight Selfie Generator pipeline

Tests the complete pipeline with sample images
"""

import os
import sys
from pathlib import Path
import argparse
from typing import List

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent / "backend"))

from pipeline import CompleteInflightSelfiePipeline


def test_pipeline(
    person1_images: List[str],
    person2_images: List[str],
    prompt: str = "Two friends taking a sunset selfie flying to Dubai",
    output_path: str = "test_output.png",
    models_dir: str = "./models",
    use_scene_planner: bool = True,
    seed: int = 42,
):
    """
    Test the complete pipeline.

    Args:
        person1_images: List of image paths for person 1
        person2_images: List of image paths for person 2
        prompt: Scene description
        output_path: Where to save result
        models_dir: Models directory
        use_scene_planner: Whether to use TinyLlama
        seed: Random seed
    """

    print("🧪 Testing Inflight Selfie Generator Pipeline")
    print("=" * 60)

    # Validate inputs
    for i, img in enumerate(person1_images):
        if not os.path.exists(img):
            print(f"❌ Error: Person 1 image {i+1} not found: {img}")
            return False

    for i, img in enumerate(person2_images):
        if not os.path.exists(img):
            print(f"❌ Error: Person 2 image {i+1} not found: {img}")
            return False

    print(f"\n📋 Configuration:")
    print(f"  Prompt: {prompt}")
    print(f"  Person 1 images: {len(person1_images)}")
    print(f"  Person 2 images: {len(person2_images)}")
    print(f"  Models dir: {models_dir}")
    print(f"  Scene planner: {'enabled' if use_scene_planner else 'disabled'}")
    print(f"  Seed: {seed}")
    print(f"  Output: {output_path}")

    try:
        # Initialize pipeline
        print("\n🚀 Initializing pipeline...")
        pipeline = CompleteInflightSelfiePipeline(
            models_dir=models_dir,
            use_scene_planner=use_scene_planner,
        )

        # Generate
        print("\n🎨 Generating selfie...")
        result = pipeline.generate(
            user_prompt=prompt,
            person1_images=person1_images,
            person2_images=person2_images,
            seed=seed,
        )

        # Save
        print(f"\n💾 Saving result to: {output_path}")
        result.save(output_path)

        print("\n" + "=" * 60)
        print("✅ Test completed successfully!")
        print(f"📸 Output saved to: {output_path}")
        print("=" * 60)

        return True

    except Exception as e:
        print("\n" + "=" * 60)
        print(f"❌ Test failed: {e}")
        print("=" * 60)
        import traceback
        traceback.print_exc()
        return False


def main():
    parser = argparse.ArgumentParser(description="Test Inflight Selfie Generator pipeline")

    parser.add_argument(
        "--person1",
        nargs="+",
        required=True,
        help="Image paths for person 1 (1-5 images)"
    )
    parser.add_argument(
        "--person2",
        nargs="+",
        required=True,
        help="Image paths for person 2 (1-5 images)"
    )
    parser.add_argument(
        "--prompt",
        type=str,
        default="Two friends taking a sunset selfie flying to Dubai",
        help="Scene description"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="test_output.png",
        help="Output image path"
    )
    parser.add_argument(
        "--models-dir",
        type=str,
        default="./models",
        help="Models directory"
    )
    parser.add_argument(
        "--no-scene-planner",
        action="store_true",
        help="Disable TinyLlama scene planner"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility"
    )

    args = parser.parse_args()

    success = test_pipeline(
        person1_images=args.person1,
        person2_images=args.person2,
        prompt=args.prompt,
        output_path=args.output,
        models_dir=args.models_dir,
        use_scene_planner=not args.no_scene_planner,
        seed=args.seed,
    )

    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
