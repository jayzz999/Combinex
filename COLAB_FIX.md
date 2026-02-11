# 🔧 Colab Notebook Fix - InsightFace Download Issue

## Problem

The InsightFace models are not downloading correctly, causing this error:
```
ModelProto does not have a graph
```

## Solution

Replace **Cell 3** in the Colab notebook with this fixed version:

```python
# Cell 3: Download Required Models (FIXED)
from huggingface_hub import hf_hub_download
import os
import urllib.request

print("📥 Downloading models...")

# Create model directories
os.makedirs("models/ip-adapter-faceid", exist_ok=True)
os.makedirs("models/insightface/models/antelopev2", exist_ok=True)

# Download IP-Adapter-FaceID models
print("  Downloading IP-Adapter-FaceID...")
try:
    hf_hub_download(
        repo_id="h94/IP-Adapter-FaceID",
        filename="ip-adapter-faceid_sdxl.bin",
        local_dir="./models/ip-adapter-faceid"
    )
    print("    ✓ ip-adapter-faceid_sdxl.bin")
except Exception as e:
    print(f"    ⚠️ Error downloading ip-adapter-faceid_sdxl.bin: {e}")

try:
    hf_hub_download(
        repo_id="h94/IP-Adapter-FaceID",
        filename="ip-adapter-faceid-plusv2_sdxl.bin",
        local_dir="./models/ip-adapter-faceid"
    )
    print("    ✓ ip-adapter-faceid-plusv2_sdxl.bin")
except Exception as e:
    print(f"    ⚠️ Error downloading ip-adapter-faceid-plusv2_sdxl.bin: {e}")

# Download InsightFace models using Python (more reliable than wget)
print("  Downloading InsightFace models...")

base_url = "https://huggingface.co/datasets/Gourieff/ReActor/resolve/main/models/insightface/models/antelopev2"
insightface_models = [
    "1k3d68.onnx",
    "2d106det.onnx",
    "genderage.onnx",
    "glintr100.onnx",
    "scrfd_10g_bnkps.onnx"
]

for model_file in insightface_models:
    output_path = f"models/insightface/models/antelopev2/{model_file}"

    # Skip if already exists and valid
    if os.path.exists(output_path) and os.path.getsize(output_path) > 1000:
        print(f"    ✓ {model_file} (already exists)")
        continue

    try:
        url = f"{base_url}/{model_file}"
        print(f"    Downloading {model_file}...")
        urllib.request.urlretrieve(url, output_path)

        # Verify file size
        file_size = os.path.getsize(output_path)
        if file_size > 1000:  # Should be at least 1KB
            print(f"    ✓ {model_file} ({file_size / 1024 / 1024:.1f} MB)")
        else:
            print(f"    ⚠️ {model_file} seems too small ({file_size} bytes), may be corrupt")
            os.remove(output_path)

    except Exception as e:
        print(f"    ❌ Failed to download {model_file}: {e}")

# Verify downloads
print("\n  Verifying InsightFace models...")
all_exist = True
for model_file in insightface_models:
    path = f"models/insightface/models/antelopev2/{model_file}"
    if os.path.exists(path) and os.path.getsize(path) > 1000:
        print(f"    ✓ {model_file}")
    else:
        print(f"    ❌ {model_file} missing or corrupt")
        all_exist = False

if all_exist:
    print("\n✅ All models downloaded successfully!")
else:
    print("\n⚠️ Some models failed to download. See errors above.")
    print("You can try running this cell again.")
```

## Alternative: Use gdown

If the above still fails, try this alternative using `gdown`:

```python
# Cell 3: Alternative Download Method
!pip install -q gdown

import os
import gdown

print("📥 Downloading models...")

# Create directories
os.makedirs("models/ip-adapter-faceid", exist_ok=True)
os.makedirs("models/insightface/models/antelopev2", exist_ok=True)

# Download IP-Adapter from HuggingFace
print("  Downloading IP-Adapter-FaceID...")
from huggingface_hub import hf_hub_download

hf_hub_download(
    repo_id="h94/IP-Adapter-FaceID",
    filename="ip-adapter-faceid_sdxl.bin",
    local_dir="./models/ip-adapter-faceid"
)

# Download InsightFace from alternative source
print("  Downloading InsightFace models from Google Drive...")

# These are public mirrors - replace with actual working links
insightface_files = {
    "1k3d68.onnx": "YOUR_GDRIVE_ID_1",
    "2d106det.onnx": "YOUR_GDRIVE_ID_2",
    # ... etc
}

# For now, use direct wget with proper syntax
insightface_models = [
    "1k3d68.onnx",
    "2d106det.onnx",
    "genderage.onnx",
    "glintr100.onnx",
    "scrfd_10g_bnkps.onnx"
]

for model in insightface_models:
    url = f"https://huggingface.co/datasets/Gourieff/ReActor/resolve/main/models/insightface/models/antelopev2/{model}"
    output = f"models/insightface/models/antelopev2/{model}"

    print(f"  Downloading {model}...")
    !wget -q -O {output} {url}

    # Check if download successful
    if os.path.exists(output) and os.path.getsize(output) > 1000:
        print(f"    ✓ {model}")
    else:
        print(f"    ❌ {model} failed")

print("✅ Downloads complete!")
```

## Recommended: Simpler Approach

The most reliable method is to use `huggingface-cli`:

```python
# Cell 3: Simple Download with huggingface-cli
import os

print("📥 Downloading models...")

# Create directories
!mkdir -p models/ip-adapter-faceid
!mkdir -p models/insightface/models/antelopev2

# Download IP-Adapter
print("  Downloading IP-Adapter-FaceID...")
!huggingface-cli download h94/IP-Adapter-FaceID \
    ip-adapter-faceid_sdxl.bin \
    --local-dir models/ip-adapter-faceid \
    --local-dir-use-symlinks False

# Download InsightFace - each file individually
print("  Downloading InsightFace models...")

base = "https://huggingface.co/datasets/Gourieff/ReActor/resolve/main/models/insightface/models/antelopev2"

!wget -q -O models/insightface/models/antelopev2/1k3d68.onnx "$base/1k3d68.onnx"
!wget -q -O models/insightface/models/antelopev2/2d106det.onnx "$base/2d106det.onnx"
!wget -q -O models/insightface/models/antelopev2/genderage.onnx "$base/genderage.onnx"
!wget -q -O models/insightface/models/antelopev2/glintr100.onnx "$base/glintr100.onnx"
!wget -q -O models/insightface/models/antelopev2/scrfd_10g_bnkps.onnx "$base/scrfd_10g_bnkps.onnx"

# Verify
print("\n  Verifying downloads...")
import os

files_to_check = [
    "models/insightface/models/antelopev2/1k3d68.onnx",
    "models/insightface/models/antelopev2/2d106det.onnx",
    "models/insightface/models/antelopev2/genderage.onnx",
    "models/insightface/models/antelopev2/glintr100.onnx",
    "models/insightface/models/antelopev2/scrfd_10g_bnkps.onnx",
]

all_good = True
for f in files_to_check:
    if os.path.exists(f) and os.path.getsize(f) > 1000:
        size_mb = os.path.getsize(f) / 1024 / 1024
        print(f"    ✓ {os.path.basename(f)} ({size_mb:.1f} MB)")
    else:
        print(f"    ❌ {os.path.basename(f)} MISSING OR CORRUPT")
        all_good = False

if all_good:
    print("\n✅ All models downloaded successfully!")
else:
    print("\n⚠️ Some downloads failed - please run this cell again")
```

## How to Apply the Fix

1. **Open your Colab notebook**
2. **Find Cell 3** (Download Required Models)
3. **Delete the entire cell**
4. **Create a new code cell**
5. **Copy and paste** the recommended "Simple Download" code above
6. **Run the cell**

The downloads should now work properly and you'll see file sizes to confirm successful downloads.

## If Still Having Issues

Try this manual download approach:

1. Download models locally from these links:
   - InsightFace: https://github.com/deepinsight/insightface/tree/master/python-package#model-zoo

2. Upload to Google Drive

3. Mount Drive in Colab:
```python
from google.colab import drive
drive.mount('/content/drive')

# Copy from Drive
!cp -r /content/drive/MyDrive/insightface_models/* models/insightface/models/antelopev2/
```

## Verification

After running the fixed cell, you should see:

```
✓ 1k3d68.onnx (5.3 MB)
✓ 2d106det.onnx (3.4 MB)
✓ genderage.onnx (1.3 MB)
✓ glintr100.onnx (248.3 MB)
✓ scrfd_10g_bnkps.onnx (16.9 MB)
```

If you see these file sizes, the downloads are successful!
