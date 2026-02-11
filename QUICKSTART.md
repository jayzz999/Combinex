# 🚀 Quick Start Guide

Get started with the Inflight Selfie Generator in 3 steps!

## Prerequisites

- **For Training**: Google Account (FREE Colab GPU)
- **For Production**: NVIDIA GPU with 12GB+ VRAM
- Python 3.9+
- Git

## Step 1: Training (40 minutes)

### Open Google Colab

1. Go to [Google Colab](https://colab.research.google.com)
2. Upload `notebooks/Inflight_Selfie_Training.ipynb`
3. Enable GPU: **Runtime → Change runtime type → T4 GPU**

### Run Training

Execute all cells in order:

```python
# Cell 1: Install dependencies (3 min)
# Cell 2: Verify GPU
# Cell 3: Download models (5 min)
# Cell 4: Initialize pipeline (2 min)
# Cell 5: Scene planning data
# Cell 6: Fine-tune TinyLlama (20 min)
# Cell 7: Complete pipeline
# Cell 8: Test generation
# Cell 9: Gradio demo (optional)
# Cell 10: Export models
```

### Download Models

At the end, download `inflight_selfie_models.zip` (~50MB)

**Total Time**: ~40 minutes

---

## Step 2: Local Setup (5 minutes)

### Clone and Install

```bash
# Clone repository
git clone <your-repo-url>
cd inflight-selfie-generator

# Create virtual environment
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# Install dependencies
pip install -r backend/requirements.txt

# Download base models
python scripts/download_models.py
```

### Add Fine-tuned Models

```bash
# Extract Colab models
unzip inflight_selfie_models.zip -d backend/models/
```

---

## Step 3: Run! (2 minutes)

### Start Server

```bash
cd backend
uvicorn server:app --host 0.0.0.0 --port 8000
```

Wait for initialization (~30 seconds)

### Test API

Visit http://localhost:8000/docs for interactive documentation

Or test with curl:

```bash
curl -X POST "http://localhost:8000/generate" \
  -F "prompt=Two friends sunset selfie flying to Dubai" \
  -F "person1_images=@person1.jpg" \
  -F "person2_images=@person2.jpg"
```

### Python Usage

```python
from pipeline import generate_inflight_selfie

result = generate_inflight_selfie(
    user_prompt="sunset selfie flying to Dubai",
    person1_images=["person1_1.jpg", "person1_2.jpg"],
    person2_images=["person2_1.jpg", "person2_2.jpg"],
    output_path="output.png"
)
```

---

## ✅ Success!

You should now have:
- ✅ Running FastAPI server at http://localhost:8000
- ✅ Interactive docs at http://localhost:8000/docs
- ✅ Working pipeline for Python scripts

---

## Next Steps

1. **Test with your photos**: Upload 2-5 face images per person
2. **Try different prompts**: See [README.md](README.md#example-prompts)
3. **Optimize parameters**: Adjust `ip_adapter_scale` and `guidance_scale`
4. **Build frontend**: Create Next.js UI (optional)

---

## Troubleshooting

### GPU Not Found

```bash
# Check CUDA
python -c "import torch; print(torch.cuda.is_available())"

# If False, install CUDA 11.8+
```

### Out of Memory

```bash
# Enable CPU offloading
export CPU_OFFLOAD=true
```

### Face Not Detected

- Use clear, well-lit photos
- Face should be visible (no sunglasses/masks)
- Try different images

### Slow Generation

- Reduce inference steps: `num_inference_steps=20`
- Use smaller resolution (but quality drops)

---

## Getting Help

- **API Docs**: http://localhost:8000/docs
- **Health Check**: http://localhost:8000/health
- **Full README**: [README.md](README.md)
- **Issues**: Create GitHub issue

---

## Example Prompts to Try

| Prompt | Expected Result |
|--------|----------------|
| `"Two friends sunset selfie flying to Dubai"` | Golden sunset through window |
| `"Business class champagne celebration"` | Luxury cabin with champagne |
| `"Night flight with city lights below"` | Dark cabin, city glow |
| `"Morning flight over fluffy clouds"` | Bright, cheerful morning |
| `"Excited takeoff selfie, runway visible"` | Dynamic departure moment |

Enjoy creating amazing inflight selfies! ✈️
