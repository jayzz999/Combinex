"""
Test script for FastAPI server

Tests all API endpoints
"""

import requests
import argparse
import base64
from pathlib import Path
from typing import List


def test_health(base_url: str):
    """Test health endpoint."""
    print("\n🏥 Testing /health endpoint...")

    try:
        response = requests.get(f"{base_url}/health")
        response.raise_for_status()

        data = response.json()
        print(f"  Status: {data['status']}")
        print(f"  Pipeline loaded: {data['pipeline_loaded']}")

        if data['gpu']['available']:
            print(f"  GPU: {data['gpu']['name']}")
            print(f"  Memory allocated: {data['gpu']['memory_allocated_gb']:.2f} GB")
        else:
            print("  GPU: Not available (CPU mode)")

        return True

    except Exception as e:
        print(f"  ❌ Error: {e}")
        return False


def test_examples(base_url: str):
    """Test examples endpoint."""
    print("\n📚 Testing /examples endpoint...")

    try:
        response = requests.get(f"{base_url}/examples")
        response.raise_for_status()

        data = response.json()
        examples = data['examples']

        print(f"  Found {len(examples)} example prompts:")
        for ex in examples[:3]:  # Show first 3
            print(f"    - {ex['title']}: {ex['prompt'][:60]}...")

        return True

    except Exception as e:
        print(f"  ❌ Error: {e}")
        return False


def test_generate(
    base_url: str,
    person1_images: List[str],
    person2_images: List[str],
    prompt: str,
    output_path: str = "api_test_output.png",
    seed: int = 42,
):
    """Test generate endpoint."""
    print("\n🎨 Testing /generate endpoint...")

    try:
        # Prepare files
        files = []
        for img_path in person1_images:
            files.append(('person1_images', open(img_path, 'rb')))

        for img_path in person2_images:
            files.append(('person2_images', open(img_path, 'rb')))

        # Prepare data
        data = {
            'prompt': prompt,
            'num_inference_steps': 30,
            'seed': seed,
        }

        print(f"  Prompt: {prompt}")
        print(f"  Person 1: {len(person1_images)} images")
        print(f"  Person 2: {len(person2_images)} images")
        print(f"  Generating...")

        # Send request
        response = requests.post(
            f"{base_url}/generate",
            files=files,
            data=data,
            timeout=300,  # 5 minutes
        )

        # Close files
        for _, f in files:
            f.close()

        response.raise_for_status()

        # Parse response
        result = response.json()

        if result['success']:
            print(f"  ✅ Generation successful!")
            print(f"  Prompt used: {result['prompt_used'][:60]}...")
            print(f"  Parameters: {result['parameters']}")

            # Save image
            if 'image' in result:
                # Decode base64
                img_data = result['image'].split(',')[1]  # Remove data:image/png;base64,
                img_bytes = base64.b64decode(img_data)

                with open(output_path, 'wb') as f:
                    f.write(img_bytes)

                print(f"  💾 Saved to: {output_path}")

            return True
        else:
            print(f"  ❌ Generation failed: {result.get('error', 'Unknown error')}")
            return False

    except requests.exceptions.Timeout:
        print(f"  ❌ Request timed out (>5 minutes)")
        return False

    except Exception as e:
        print(f"  ❌ Error: {e}")
        return False


def test_all(
    base_url: str,
    person1_images: List[str] = None,
    person2_images: List[str] = None,
    prompt: str = "Two friends taking sunset selfie flying to Dubai",
    output_path: str = "api_test_output.png",
):
    """Run all tests."""

    print("🧪 Testing Inflight Selfie Generator API")
    print("=" * 60)
    print(f"Base URL: {base_url}")

    results = {}

    # Test health
    results['health'] = test_health(base_url)

    # Test examples
    results['examples'] = test_examples(base_url)

    # Test generate (if images provided)
    if person1_images and person2_images:
        results['generate'] = test_generate(
            base_url,
            person1_images,
            person2_images,
            prompt,
            output_path,
        )
    else:
        print("\n⚠️ Skipping /generate test (no images provided)")
        print("   Use --person1 and --person2 to test generation")

    # Summary
    print("\n" + "=" * 60)
    print("📊 Test Summary:")
    for test, passed in results.items():
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"  {test}: {status}")

    all_passed = all(results.values())
    print("\n" + ("✅ All tests passed!" if all_passed else "❌ Some tests failed"))
    print("=" * 60)

    return all_passed


def main():
    parser = argparse.ArgumentParser(description="Test Inflight Selfie Generator API")

    parser.add_argument(
        "--url",
        type=str,
        default="http://localhost:8000",
        help="Base URL of API server"
    )
    parser.add_argument(
        "--person1",
        nargs="+",
        help="Image paths for person 1 (optional, for generation test)"
    )
    parser.add_argument(
        "--person2",
        nargs="+",
        help="Image paths for person 2 (optional, for generation test)"
    )
    parser.add_argument(
        "--prompt",
        type=str,
        default="Two friends taking sunset selfie flying to Dubai",
        help="Scene description for generation test"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="api_test_output.png",
        help="Output path for generated image"
    )

    args = parser.parse_args()

    success = test_all(
        base_url=args.url,
        person1_images=args.person1,
        person2_images=args.person2,
        prompt=args.prompt,
        output_path=args.output,
    )

    import sys
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
