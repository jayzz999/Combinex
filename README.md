# Inflight Selfie Generator

**AI-Powered Realistic Airplane Cabin Selfies with Face Identity Preservation**

![Python](https://img.shields.io/badge/Python-3.9+-3776AB?logo=python&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-EE4C2C?logo=pytorch&logoColor=white)
![FastAPI](https://img.shields.io/badge/FastAPI-0.108+-009688?logo=fastapi&logoColor=white)
![Stable Diffusion](https://img.shields.io/badge/SDXL-1.0-7C3AED)
![IP-Adapter](https://img.shields.io/badge/IP--Adapter-FaceID-blue)
![Colab](https://img.shields.io/badge/Training-Google_Colab-F9AB00?logo=googlecolab&logoColor=white)
![License](https://img.shields.io/badge/License-MIT-green)

Generate realistic inflight selfies of two people using **IP-Adapter-FaceID** for face-preserving synthesis, **Stable Diffusion XL** for high-quality image generation, and a **TinyLlama** scene planner that automatically optimizes generation parameters from natural language prompts. Train for free on Google Colab in ~40 minutes.

---

## How It Works

```
User Prompt (natural language)
         |
    TinyLlama Scene Planner --> Optimal generation parameters
         |
  Face Images (Person 1 & 2)
         |
    InsightFace --> 512-d face embeddings
         |
  IP-Adapter-FaceID + SDXL --> 1024x1024 realistic inflight selfie
```

1. **Scene Planning** — TinyLlama (fine-tuned with LoRA) converts your prompt into optimal `ip_adapter_scale`, `guidance_scale`, and `num_inference_steps`
2. **Face Embedding** — InsightFace (antelopev2) extracts 512-dimensional face embeddings from 1-5 reference photos per person
3. **Image Generation** — Stable Diffusion XL + IP-Adapter-FaceID generates a 1024x1024 selfie preserving both identities in the described scene

---

## Features

- **Face Identity Preservation** — IP-Adapter-FaceID maintains recognizable likeness of both people
- **Intelligent Scene Planning** — TinyLlama auto-optimizes parameters from natural language prompts
- **Multi-Person Support** — Generate selfies with two people from separate reference photos
- **Multiple Scenes** — Sunset flights, business class, night flights, tropical destinations, takeoff, and more
- **Production-Ready API** — FastAPI backend with OpenAPI docs, health checks, and streaming endpoints
- **Free Training** — Fine-tune on Google Colab T4 GPU in ~40 minutes, no local GPU needed
- **Reproducible** — Seed support for deterministic generation
- **CPU Offloading** — Run on GPUs with limited VRAM by offloading to system RAM

---

## Quick Start

### Step 1: Train on Google Colab (~40 min)

1. Upload `notebooks/Inflight_Selfie_Training.ipynb` to [Google Colab](https://colab.research.google.com)
2. Enable GPU: **Runtime** → **Change runtime type** → **T4 GPU**
3. Run all cells sequentially
4. Download `inflight_selfie_models.zip` from the final cell

### Step 2: Set Up Backend

```bash
git clone https://github.com/jayzz999/Combinex.git
cd Combinex/inflight-selfie-generator

python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

cd backend
pip install -r requirements.txt

# Extract trained models from Colab
unzip inflight_selfie_models.zip -d models/
```

### Step 3: Configure

```bash
cp .env.example .env
# Edit .env to set DEVICE=cuda (or cpu), MODELS_DIR, etc.
```

### Step 4: Run

```bash
uvicorn server:app --host 0.0.0.0 --port 8000
```

- **API Docs**: http://localhost:8000/docs
- **Health Check**: http://localhost:8000/health

---

## Usage

### Python API

```python
from pipeline import generate_inflight_selfie

result = generate_inflight_selfie(
    user_prompt="Two friends taking sunset selfie flying to Dubai",
    person1_images=["person1_face1.jpg", "person1_face2.jpg"],
    person2_images=["person2_face1.jpg", "person2_face2.jpg"],
    output_path="output.png",
    seed=42
)
```

### REST API

```bash
curl -X POST "http://localhost:8000/generate" \
  -F "prompt=Two friends sunset selfie flying to Dubai" \
  -F "person1_images=@person1.jpg" \
  -F "person2_images=@person2.jpg" \
  -F "num_inference_steps=30"
```

### API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/` | API info |
| GET | `/health` | GPU status, memory usage, pipeline status |
| GET | `/examples` | Example prompts |
| POST | `/generate` | Generate selfie (returns JSON + base64 image) |
| POST | `/generate/stream` | Generate selfie (returns streaming PNG) |

### Request Parameters

| Parameter | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| `prompt` | string | Yes | -- | Scene description |
| `person1_images` | file(s) | Yes | -- | 1-5 face images of person 1 |
| `person2_images` | file(s) | Yes | -- | 1-5 face images of person 2 |
| `num_inference_steps` | int | No | 30 | Quality steps (20-50) |
| `seed` | int | No | random | For reproducibility |
| `ip_adapter_scale` | float | No | auto | Face similarity (0.0-1.0) |
| `guidance_scale` | float | No | auto | Prompt adherence (1.0-20.0) |

---

## Example Prompts

| Scene | Prompt |
|-------|--------|
| Sunset Flight | "Two friends taking sunset selfie flying to Dubai, window seat, golden hour" |
| Business Class | "Business class celebration with champagne, luxury cabin" |
| Night Flight | "Night flight selfie with city lights below, dim cabin lighting" |
| Tropical | "Selfie flying over tropical ocean and islands, vacation excitement" |
| First Class | "First class suite selfie, spacious luxury cabin with premium amenities" |
| Takeoff | "Excited selfie during takeoff with runway visible through window" |

---

## Project Structure

```
inflight-selfie-generator/
├── backend/
│   ├── pipeline.py                # Core pipeline (3 classes: InflightSelfiePipeline,
│   │                              #   ScenePlanner, CompleteInflightSelfiePipeline)
│   ├── server.py                  # FastAPI REST API server
│   ├── requirements.txt           # Production dependencies
│   └── .env.example               # Configuration template
│
├── notebooks/
│   └── Inflight_Selfie_Training.ipynb  # Google Colab training notebook
│
├── scripts/
│   ├── download_models.py         # Base model downloader
│   ├── prepare_dataset.py         # Dataset preparation utility
│   ├── test_pipeline.py           # Pipeline integration test
│   └── test_api.py                # API endpoint test
│
├── models/                        # Trained models (downloaded after Colab training)
├── dataset/                       # Training data directory
│   ├── images/                    # Target selfie images
│   └── face_refs/                 # Face reference images
├── frontend/                      # Optional Next.js UI (planned)
│
├── setup.sh                       # Automated setup script
├── requirements-colab.txt         # Colab-specific dependencies
└── README.md
```

---

## Tech Stack

| Component | Technology | Details |
|-----------|-----------|---------|
| Face Encoding | InsightFace (antelopev2) | 512-dimensional face embeddings |
| Image Generation | Stable Diffusion XL 1.0 | 1024x1024 output resolution |
| Face Preservation | IP-Adapter-FaceID | Face-guided diffusion |
| Scene Planner | TinyLlama 1.1B Chat | LoRA fine-tuned (r=16) |
| Training | Unsloth + LoRA | 2x faster on Colab T4 |
| Backend | FastAPI + Uvicorn | Async REST API |
| Deep Learning | PyTorch 2.0+ | CUDA + CPU offloading |
| Model Hub | Hugging Face | diffusers, transformers, peft |

---

## Performance

| Hardware | Generation Time | VRAM Usage |
|----------|----------------|------------|
| RTX 4090 | ~8-10s | 10 GB |
| RTX 3080 (12GB) | ~12-15s | 11 GB |
| T4 (Colab) | ~20-25s | 12 GB |
| CPU (16-core) | ~5-8 min | 8 GB RAM |

*Based on 30 inference steps at 1024x1024 resolution*

### System Requirements

**Minimum (GPU)**: NVIDIA GPU with 12GB+ VRAM, 16GB RAM, 30GB storage, CUDA 11.8+

**Recommended**: RTX 3080+ / A10 / T4, 32GB RAM, SSD

**CPU-only**: Supported but slow (5-10 min per image)

---

## Training Details

| Spec | Value |
|------|-------|
| Platform | Google Colab (Free T4 GPU) |
| Fine-tuning Method | LoRA (r=16) via Unsloth |
| Base Model | TinyLlama 1.1B Chat |
| Training Time | ~20 min (fine-tuning) + ~20 min (setup) |
| Total Models Size | ~30 GB |

### Training Breakdown

| Cell | Task | Time |
|------|------|------|
| 1 | Install dependencies | ~3 min |
| 2-4 | Model setup | ~5-7 min |
| 5-6 | TinyLlama LoRA fine-tuning | ~20 min |
| 7-8 | Test generation | ~5 min |
| 9 | Export models | ~2 min |

---

## Configuration

Create `backend/.env` from the template:

```bash
MODELS_DIR=./models         # Path to trained models
DEVICE=cuda                 # cuda or cpu
CPU_OFFLOAD=true            # Offload to RAM if VRAM is limited
USE_SCENE_PLANNER=true      # Enable TinyLlama scene optimization
HOST=0.0.0.0
PORT=8000
DEBUG=false
LOG_LEVEL=info
```

---

## Best Practices

### Face Images
- Use **2-5 clear, well-lit** photos per person
- Frontal or slight angle works best
- Minimum resolution: 512x512
- Avoid sunglasses or face obstructions

### Prompts
- Be specific about the scene (time of day, location, mood)
- Include cabin details (window seat, business class)
- Mention expressions (happy, excited, relaxed)

### Parameter Tuning
- `ip_adapter_scale` 0.6-0.7: Balanced identity and scene quality
- `guidance_scale` 7.0-8.0: Good prompt adherence
- `num_inference_steps` 30-40: Quality vs speed tradeoff

---

## Troubleshooting

| Issue | Solution |
|-------|----------|
| GPU out of memory | Set `CPU_OFFLOAD=true` in `.env` |
| Face not detected | Use clearer photos, check resolution |
| Poor identity preservation | Use more reference images (3-5), increase `ip_adapter_scale` to 0.7-0.8 |
| Scene doesn't match prompt | Increase `guidance_scale` to 8-10, be more specific |
| Slow generation | Reduce `num_inference_steps` to 20-25, ensure GPU is active |
| InsightFace Colab issues | See `COLAB_INSIGHTFACE_FIX.md` |

---

## Roadmap

- [x] IP-Adapter-FaceID integration
- [x] TinyLlama scene planner (LoRA fine-tuned)
- [x] FastAPI backend with REST API
- [x] Google Colab training notebook
- [x] CPU offloading support
- [ ] Next.js frontend UI
- [ ] Multi-face improved compositing
- [ ] ControlNet pose guidance
- [ ] Background replacement
- [ ] Batch generation API
- [ ] Docker deployment
- [ ] Cloud hosting guide

---

## License

MIT License

## Credits

- [IP-Adapter-FaceID](https://github.com/tencent-ailab/IP-Adapter) — Tencent AI Lab
- [Stable Diffusion XL](https://github.com/Stability-AI/generative-models) — Stability AI
- [InsightFace](https://github.com/deepinsight/insightface) — Jia Guo, Jiankang Deng
- [TinyLlama](https://github.com/jzhang38/TinyLlama) — Zhang et al.
- [Unsloth](https://github.com/unslothai/unsloth) — Daniel Han, Michael Han
