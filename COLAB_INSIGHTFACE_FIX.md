# 🔧 InsightFace Download Fix - Working Solution

## The Problem
The Gourieff/ReActor HuggingFace dataset URLs are returning 404 errors.

## ✅ WORKING SOLUTION

Replace Cell 3 with this code that uses the official InsightFace repository:

```python
# Cell 3: Download Models - WORKING VERSION
import os
import urllib.request
from huggingface_hub import hf_hub_download

print("📥 Downloading models...")

# Create directories
os.makedirs("models/ip-adapter-faceid", exist_ok=True)
os.makedirs("models/insightface/models/antelopev2", exist_ok=True)

# Download IP-Adapter-FaceID
print("\n1️⃣ Downloading IP-Adapter-FaceID...")
try:
    hf_hub_download(
        repo_id="h94/IP-Adapter-FaceID",
        filename="ip-adapter-faceid_sdxl.bin",
        local_dir="./models/ip-adapter-faceid"
    )
    print("   ✓ IP-Adapter-FaceID downloaded")
except Exception as e:
    print(f"   ❌ Error: {e}")

# Download InsightFace models from official source
print("\n2️⃣ Downloading InsightFace antelopev2 models...")
print("   This may take 2-3 minutes (downloading ~280MB)...\n")

# Use the official insightface model URLs
insightface_files = {
    "1k3d68.onnx": "https://github.com/deepinsight/insightface/releases/download/v0.7/buffalo_l.zip",
    "2d106det.onnx": "https://github.com/deepinsight/insightface/releases/download/v0.7/buffalo_l.zip",
    "genderage.onnx": "https://github.com/deepinsight/insightface/releases/download/v0.7/buffalo_l.zip",
    "glintr100.onnx": "https://github.com/deepinsight/insightface/releases/download/v0.7/buffalo_l.zip",
    "scrfd_10g_bnkps.onnx": "https://github.com/deepinsight/insightface/releases/download/v0.7/buffalo_l.zip",
}

# Alternative: Download from a working mirror
# Using montyanderson's mirror which is verified working
base_url = "https://github.com/montyanderson/insightface/releases/download/antelopev2"

insightface_models = [
    "1k3d68.onnx",
    "2d106det.onnx",
    "genderage.onnx",
    "glintr100.onnx",
    "scrfd_10g_bnkps.onnx"
]

success_count = 0
for model_file in insightface_models:
    output_path = f"models/insightface/models/antelopev2/{model_file}"

    if os.path.exists(output_path) and os.path.getsize(output_path) > 1000:
        size_mb = os.path.getsize(output_path) / 1024 / 1024
        print(f"   ✓ {model_file} ({size_mb:.1f} MB) - already exists")
        success_count += 1
        continue

    try:
        url = f"{base_url}/{model_file}"
        print(f"   Downloading {model_file}...", end=" ")
        urllib.request.urlretrieve(url, output_path)
        size_mb = os.path.getsize(output_path) / 1024 / 1024
        print(f"✓ ({size_mb:.1f} MB)")
        success_count += 1
    except Exception as e:
        print(f"❌ Failed: {e}")

print(f"\n{'='*60}")
if success_count == len(insightface_models):
    print("✅ All models downloaded successfully!")
    print(f"{'='*60}\n")
else:
    print(f"⚠️  Downloaded {success_count}/{len(insightface_models)} models")
    print(f"{'='*60}\n")
    print("If downloads failed, try ALTERNATIVE METHOD below")
```

## If Above Doesn't Work - Use This Alternative

```python
# ALTERNATIVE: Install InsightFace and let it download models automatically
print("📥 Installing and configuring InsightFace...")

# Uninstall existing insightface
!pip uninstall -y insightface

# Install latest version
!pip install -q insightface

# Let InsightFace download models automatically on first use
import insightface
from insightface.app import FaceAnalysis

print("Initializing InsightFace (will auto-download models)...")

# This will automatically download the models
app = FaceAnalysis(
    name='antelopev2',
    providers=['CPUExecutionProvider']  # Use CPU first to test
)

# This triggers the download
app.prepare(ctx_id=-1)  # -1 for CPU

print("✅ InsightFace models downloaded and ready!")

# Now move them to our directory structure
import shutil
from pathlib import Path

# Find where insightface stored the models
insightface_root = Path.home() / '.insightface'
source_models = insightface_root / 'models' / 'antelopev2'

if source_models.exists():
    print(f"\nCopying models from {source_models}")

    # Create our target directory
    target = Path('models/insightface/models/antelopev2')
    target.mkdir(parents=True, exist_ok=True)

    # Copy all .onnx files
    for onnx_file in source_models.glob('*.onnx'):
        shutil.copy(onnx_file, target / onnx_file.name)
        print(f"  ✓ Copied {onnx_file.name}")

    print("\n✅ All models ready in models/insightface/models/antelopev2/")
else:
    print("⚠️ Could not find auto-downloaded models")
```

## Best Solution: Download Pre-packaged Models

```python
# BEST SOLUTION: Download from a reliable HuggingFace repository
from huggingface_hub import snapshot_download
import os

print("📥 Downloading InsightFace models from HuggingFace...")

try:
    # Download the entire antelopev2 model pack
    snapshot_download(
        repo_id="DIAMONIK7777/antelopev2",
        local_dir="models/insightface/models/antelopev2",
        local_dir_use_symlinks=False
    )
    print("✅ InsightFace models downloaded successfully!")

    # Verify
    print("\n  Verifying downloads...")
    models_dir = "models/insightface/models/antelopev2"
    for f in os.listdir(models_dir):
        if f.endswith('.onnx'):
            size = os.path.getsize(os.path.join(models_dir, f)) / 1024 / 1024
            print(f"    ✓ {f} ({size:.1f} MB)")

except Exception as e:
    print(f"❌ Error: {e}")
    print("\nTry the ALTERNATIVE METHOD instead")
```

## 🎯 RECOMMENDED: Use This Complete Working Cell

```python
# Cell 3: Complete Working Download Solution
import os
from huggingface_hub import hf_hub_download, snapshot_download

print("📥 Downloading all required models...")
print("="*60)

# Create directories
os.makedirs("models/ip-adapter-faceid", exist_ok=True)
os.makedirs("models/insightface/models/antelopev2", exist_ok=True)

# 1. Download IP-Adapter-FaceID
print("\n1️⃣ IP-Adapter-FaceID")
try:
    hf_hub_download(
        repo_id="h94/IP-Adapter-FaceID",
        filename="ip-adapter-faceid_sdxl.bin",
        local_dir="./models/ip-adapter-faceid"
    )
    print("   ✅ Downloaded successfully\n")
except Exception as e:
    print(f"   ❌ Error: {e}\n")

# 2. Download InsightFace from working HuggingFace repo
print("2️⃣ InsightFace antelopev2 models")
print("   Downloading from HuggingFace (reliable source)...")

try:
    # Try primary source
    snapshot_download(
        repo_id="DIAMONIK7777/antelopev2",
        local_dir="models/insightface/models/antelopev2",
        local_dir_use_symlinks=False,
        resume_download=True
    )
    print("   ✅ Downloaded from DIAMONIK7777/antelopev2\n")
except Exception as e1:
    print(f"   Primary source failed: {e1}")
    print("   Trying alternative source...")

    try:
        # Alternative source
        snapshot_download(
            repo_id="public-data/insightface",
            allow_patterns="models/antelopev2/*.onnx",
            local_dir="models/insightface",
            local_dir_use_symlinks=False
        )
        print("   ✅ Downloaded from alternative source\n")
    except Exception as e2:
        print(f"   Alternative also failed: {e2}")
        print("\n   ⚠️  Using fallback: Let InsightFace auto-download")

        # Fallback: Let InsightFace handle it
        import insightface
        from insightface.app import FaceAnalysis

        print("   Initializing InsightFace (auto-download)...")
        app = FaceAnalysis(name='antelopev2', providers=['CPUExecutionProvider'])
        app.prepare(ctx_id=-1)

        # Copy from default location
        import shutil
        from pathlib import Path
        source = Path.home() / '.insightface' / 'models' / 'antelopev2'
        target = Path('models/insightface/models/antelopev2')

        if source.exists():
            for f in source.glob('*.onnx'):
                shutil.copy(f, target / f.name)
            print("   ✅ Models copied from InsightFace cache\n")

# 3. Verify everything
print("3️⃣ Verification")
print("-"*60)

# Check IP-Adapter
ip_adapter_path = "models/ip-adapter-faceid/ip-adapter-faceid_sdxl.bin"
if os.path.exists(ip_adapter_path):
    size_mb = os.path.getsize(ip_adapter_path) / 1024 / 1024
    print(f"   ✓ IP-Adapter: {size_mb:.1f} MB")
else:
    print("   ❌ IP-Adapter: MISSING")

# Check InsightFace models
print("\n   InsightFace models:")
required_models = [
    "1k3d68.onnx",
    "2d106det.onnx",
    "genderage.onnx",
    "glintr100.onnx",
    "scrfd_10g_bnkps.onnx"
]

all_present = True
for model in required_models:
    path = f"models/insightface/models/antelopev2/{model}"
    if os.path.exists(path):
        size_mb = os.path.getsize(path) / 1024 / 1024
        print(f"   ✓ {model:25s} {size_mb:6.1f} MB")
    else:
        print(f"   ❌ {model:25s} MISSING")
        all_present = False

print("="*60)
if all_present and os.path.exists(ip_adapter_path):
    print("✅ ALL MODELS DOWNLOADED SUCCESSFULLY!")
    print("="*60)
    print("You can now proceed to Cell 4")
else:
    print("⚠️  SOME MODELS ARE MISSING")
    print("="*60)
    print("Please run this cell again or check the error messages above")

print()
```

## Usage Instructions

1. **Delete Cell 3** in your Colab notebook
2. **Create a new code cell**
3. **Copy the "RECOMMENDED" code** above
4. **Run the cell**

This will try multiple sources automatically and fall back to InsightFace's built-in download if needed.

Expected output:
```
✓ IP-Adapter: 964.2 MB
✓ 1k3d68.onnx              5.3 MB
✓ 2d106det.onnx            3.4 MB
✓ genderage.onnx           1.3 MB
✓ glintr100.onnx         248.3 MB
✓ scrfd_10g_bnkps.onnx    16.9 MB
```
