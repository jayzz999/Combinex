# API Usage Examples

Complete guide to using the Inflight Selfie Generator API.

---

## 🌐 Base URL

```
http://localhost:8000  # Local development
https://your-domain.com  # Production
```

---

## 📡 Endpoints Overview

| Endpoint | Method | Purpose |
|----------|--------|---------|
| `/` | GET | API information |
| `/health` | GET | System health status |
| `/examples` | GET | Example prompts |
| `/generate` | POST | Generate selfie (JSON response) |
| `/generate/stream` | POST | Generate selfie (image stream) |
| `/docs` | GET | Interactive API documentation |

---

## 1. Health Check

Check if the server is running and GPU status.

### cURL

```bash
curl http://localhost:8000/health
```

### Python

```python
import requests

response = requests.get("http://localhost:8000/health")
print(response.json())
```

### Response

```json
{
  "status": "healthy",
  "pipeline_loaded": true,
  "gpu": {
    "available": true,
    "name": "NVIDIA GeForce RTX 3080",
    "memory_allocated_gb": 2.3,
    "memory_reserved_gb": 3.1
  }
}
```

---

## 2. Get Example Prompts

Get inspiration for scene descriptions.

### cURL

```bash
curl http://localhost:8000/examples
```

### Python

```python
response = requests.get("http://localhost:8000/examples")
examples = response.json()["examples"]

for ex in examples:
    print(f"{ex['title']}: {ex['prompt']}")
```

### Response

```json
{
  "examples": [
    {
      "title": "Sunset Window Seat",
      "prompt": "Two friends taking a sunset selfie flying to Dubai...",
      "scene_type": "sunset_window"
    },
    ...
  ]
}
```

---

## 3. Generate Selfie (JSON Response)

Generate selfie and receive base64-encoded image in JSON.

### cURL

```bash
curl -X POST "http://localhost:8000/generate" \
  -F "prompt=Two friends taking sunset selfie flying to Dubai" \
  -F "person1_images=@person1_face1.jpg" \
  -F "person1_images=@person1_face2.jpg" \
  -F "person2_images=@person2_face1.jpg" \
  -F "num_inference_steps=30" \
  -F "seed=42"
```

### Python

```python
import requests
import base64
from PIL import Image
from io import BytesIO

# Prepare files
files = [
    ('person1_images', open('person1_face1.jpg', 'rb')),
    ('person1_images', open('person1_face2.jpg', 'rb')),
    ('person2_images', open('person2_face1.jpg', 'rb')),
]

# Request data
data = {
    'prompt': 'Two friends taking sunset selfie flying to Dubai',
    'num_inference_steps': 30,
    'seed': 42,
}

# Send request
response = requests.post(
    'http://localhost:8000/generate',
    files=files,
    data=data,
    timeout=300
)

# Close files
for _, f in files:
    f.close()

# Parse response
result = response.json()

if result['success']:
    # Decode image
    img_data = result['image'].split(',')[1]
    img_bytes = base64.b64decode(img_data)
    image = Image.open(BytesIO(img_bytes))

    # Save
    image.save('output.png')
    print("Saved to output.png")
else:
    print(f"Error: {result['error']}")
```

### JavaScript (fetch)

```javascript
async function generateSelfie() {
    const formData = new FormData();

    // Add images
    const person1File = document.getElementById('person1Input').files[0];
    const person2File = document.getElementById('person2Input').files[0];

    formData.append('person1_images', person1File);
    formData.append('person2_images', person2File);

    // Add parameters
    formData.append('prompt', 'Two friends sunset selfie flying to Dubai');
    formData.append('num_inference_steps', '30');
    formData.append('seed', '42');

    // Send request
    const response = await fetch('http://localhost:8000/generate', {
        method: 'POST',
        body: formData
    });

    const result = await response.json();

    if (result.success) {
        // Display image
        document.getElementById('output').src = result.image;
    } else {
        console.error('Error:', result.error);
    }
}
```

### Response

```json
{
  "success": true,
  "image": "data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAA...",
  "prompt_used": "two people taking selfie in airplane...",
  "parameters": {
    "num_inference_steps": 30,
    "seed": 42,
    "ip_adapter_scale": 0.65,
    "guidance_scale": 7.5
  }
}
```

---

## 4. Generate Selfie (Stream)

Generate and receive image directly (not base64).

### cURL

```bash
curl -X POST "http://localhost:8000/generate/stream" \
  -F "prompt=sunset selfie flying to Dubai" \
  -F "person1_images=@person1.jpg" \
  -F "person2_images=@person2.jpg" \
  --output result.png
```

### Python

```python
response = requests.post(
    'http://localhost:8000/generate/stream',
    files=files,
    data=data,
    stream=True
)

# Save streamed image
with open('result.png', 'wb') as f:
    for chunk in response.iter_content(chunk_size=8192):
        f.write(chunk)
```

---

## 5. Advanced Parameters

Fine-tune generation with advanced parameters.

### Python Example

```python
data = {
    'prompt': 'Business class champagne celebration',
    'num_inference_steps': 40,  # Higher = better quality
    'seed': 12345,  # For reproducibility
    'ip_adapter_scale': 0.7,  # Higher = more identity preservation
    'guidance_scale': 8.0,  # Higher = more prompt adherence
}

response = requests.post(
    'http://localhost:8000/generate',
    files=files,
    data=data
)
```

### Parameter Guide

| Parameter | Type | Range | Default | Description |
|-----------|------|-------|---------|-------------|
| `prompt` | str | - | Required | Scene description |
| `person1_images` | file[] | 1-5 | Required | Person 1 face photos |
| `person2_images` | file[] | 1-5 | Required | Person 2 face photos |
| `num_inference_steps` | int | 20-50 | 30 | Quality vs speed |
| `seed` | int | 0-∞ | None | Random seed |
| `ip_adapter_scale` | float | 0.0-1.0 | Auto | Identity strength |
| `guidance_scale` | float | 1.0-20.0 | Auto | Prompt adherence |

---

## 6. Batch Processing

Process multiple requests efficiently.

### Python

```python
import asyncio
import aiohttp

async def generate_one(session, person1, person2, prompt):
    """Generate one selfie asynchronously."""

    data = aiohttp.FormData()
    data.add_field('person1_images', open(person1[0], 'rb'))
    data.add_field('person2_images', open(person2[0], 'rb'))
    data.add_field('prompt', prompt)

    async with session.post(
        'http://localhost:8000/generate',
        data=data
    ) as response:
        return await response.json()

async def batch_generate():
    """Generate multiple selfies in sequence."""

    requests_data = [
        (['p1_1.jpg'], ['p2_1.jpg'], 'sunset selfie'),
        (['p3_1.jpg'], ['p4_1.jpg'], 'business class'),
        (['p5_1.jpg'], ['p6_1.jpg'], 'night flight'),
    ]

    async with aiohttp.ClientSession() as session:
        for i, (p1, p2, prompt) in enumerate(requests_data):
            print(f"Generating {i+1}/{len(requests_data)}...")
            result = await generate_one(session, p1, p2, prompt)

            if result['success']:
                # Save image
                img_data = result['image'].split(',')[1]
                with open(f'batch_output_{i}.png', 'wb') as f:
                    f.write(base64.b64decode(img_data))
                print(f"  ✓ Saved to batch_output_{i}.png")
            else:
                print(f"  ✗ Error: {result['error']}")

# Run batch
asyncio.run(batch_generate())
```

---

## 7. Error Handling

Handle common errors gracefully.

### Python

```python
import requests
from requests.exceptions import Timeout, ConnectionError

def safe_generate(person1_images, person2_images, prompt):
    """Generate with comprehensive error handling."""

    try:
        # Prepare request
        files = [
            ('person1_images', open(img, 'rb'))
            for img in person1_images
        ] + [
            ('person2_images', open(img, 'rb'))
            for img in person2_images
        ]

        data = {'prompt': prompt}

        # Send request
        response = requests.post(
            'http://localhost:8000/generate',
            files=files,
            data=data,
            timeout=300
        )

        # Close files
        for _, f in files:
            f.close()

        # Check response
        response.raise_for_status()
        result = response.json()

        if result['success']:
            return result
        else:
            print(f"Generation failed: {result['error']}")
            return None

    except Timeout:
        print("Request timed out (>5 minutes)")
        return None

    except ConnectionError:
        print("Could not connect to server")
        return None

    except requests.exceptions.HTTPError as e:
        print(f"HTTP error: {e}")
        return None

    except Exception as e:
        print(f"Unexpected error: {e}")
        return None
```

### Common Errors

| Status | Error | Solution |
|--------|-------|----------|
| 400 | No face detected | Use clearer face photos |
| 400 | Too many images | Max 5 per person |
| 503 | Pipeline not initialized | Wait for server startup |
| 504 | Timeout | Reduce inference steps |

---

## 8. Testing Script

Complete testing script.

### test_api.py

```python
#!/usr/bin/env python3
"""Test the API with sample requests."""

import requests
import base64
from pathlib import Path

BASE_URL = "http://localhost:8000"

def test_health():
    """Test health endpoint."""
    print("Testing /health...")
    response = requests.get(f"{BASE_URL}/health")
    print(f"  Status: {response.json()['status']}")
    return response.status_code == 200

def test_generate(person1_imgs, person2_imgs):
    """Test generation endpoint."""
    print("\nTesting /generate...")

    files = []
    for img in person1_imgs:
        files.append(('person1_images', open(img, 'rb')))
    for img in person2_imgs:
        files.append(('person2_images', open(img, 'rb')))

    data = {
        'prompt': 'Two friends sunset selfie flying to Dubai',
        'num_inference_steps': 30,
        'seed': 42,
    }

    response = requests.post(
        f"{BASE_URL}/generate",
        files=files,
        data=data,
        timeout=300
    )

    for _, f in files:
        f.close()

    if response.status_code == 200:
        result = response.json()
        if result['success']:
            # Save image
            img_data = result['image'].split(',')[1]
            with open('test_output.png', 'wb') as f:
                f.write(base64.b64decode(img_data))
            print("  ✓ Generated and saved to test_output.png")
            return True

    print(f"  ✗ Error: {response.status_code}")
    return False

if __name__ == "__main__":
    import sys

    if len(sys.argv) < 3:
        print("Usage: python test_api.py <person1.jpg> <person2.jpg>")
        sys.exit(1)

    person1 = [sys.argv[1]]
    person2 = [sys.argv[2]]

    success = test_health() and test_generate(person1, person2)
    sys.exit(0 if success else 1)
```

Usage:
```bash
python test_api.py person1.jpg person2.jpg
```

---

## 9. Integration Examples

### Flask Integration

```python
from flask import Flask, request, jsonify
import requests

app = Flask(__name__)
GENERATOR_URL = "http://localhost:8000"

@app.route('/create-selfie', methods=['POST'])
def create_selfie():
    # Forward to generator API
    response = requests.post(
        f"{GENERATOR_URL}/generate",
        files=request.files,
        data=request.form
    )
    return jsonify(response.json())
```

### Django Integration

```python
# views.py
from django.http import JsonResponse
import requests

def generate_selfie(request):
    files = {
        'person1_images': request.FILES.getlist('person1'),
        'person2_images': request.FILES.getlist('person2'),
    }

    data = {
        'prompt': request.POST.get('prompt'),
    }

    response = requests.post(
        'http://localhost:8000/generate',
        files=files,
        data=data
    )

    return JsonResponse(response.json())
```

---

## 10. Tips & Best Practices

### Image Quality
- Use high-resolution face photos (512x512+)
- Ensure faces are well-lit and clearly visible
- Avoid sunglasses, masks, or obstructions
- Upload 2-5 photos per person for consistency

### Performance
- Use `num_inference_steps=25` for faster generation
- Set shorter timeout for CPU mode
- Cache pipeline instance in production
- Use GPU for best performance

### Prompts
- Be specific about scene (time, location, mood)
- Include cabin details (window seat, business class)
- Mention expressions (happy, excited, tired)
- Reference destinations for context

### Error Handling
- Always set timeout (5+ minutes)
- Validate images before uploading
- Handle network errors gracefully
- Log errors for debugging

---

For complete API documentation, visit `/docs` when the server is running.
