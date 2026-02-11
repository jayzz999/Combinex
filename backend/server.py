"""
Inflight Selfie Generator - FastAPI Backend Server

Run with: uvicorn server:app --host 0.0.0.0 --port 8000 --reload
"""

import os
import io
import base64
import tempfile
import logging
from typing import List, Optional
from pathlib import Path

import torch
import numpy as np
from PIL import Image
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel, Field

from pipeline import CompleteInflightSelfiePipeline

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="✈️ Inflight Selfie Generator API",
    description="AI-powered inflight selfie generation using IP-Adapter-FaceID and TinyLlama",
    version="1.0.0",
)

# CORS middleware for frontend access
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global pipeline instance
pipeline: Optional[CompleteInflightSelfiePipeline] = None


# Request/Response Models
class GenerationRequest(BaseModel):
    """Request model for generation parameters."""
    prompt: str = Field(..., description="Scene description")
    num_inference_steps: int = Field(30, ge=20, le=50, description="Number of diffusion steps")
    seed: Optional[int] = Field(None, description="Random seed for reproducibility")
    ip_adapter_scale: Optional[float] = Field(None, ge=0.0, le=1.0, description="Identity preservation strength")
    guidance_scale: Optional[float] = Field(None, ge=1.0, le=20.0, description="Classifier-free guidance scale")


class GenerationResponse(BaseModel):
    """Response model for successful generation."""
    success: bool = True
    image: str = Field(..., description="Base64 encoded image data")
    prompt_used: str = Field(..., description="Actual prompt used for generation")
    parameters: dict = Field(..., description="Generation parameters used")


class ErrorResponse(BaseModel):
    """Response model for errors."""
    success: bool = False
    error: str
    detail: Optional[str] = None


# Startup and shutdown events
@app.on_event("startup")
async def startup_event():
    """Initialize the pipeline on server startup."""
    global pipeline

    logger.info("🚀 Starting Inflight Selfie Generator API...")

    try:
        # Check GPU availability
        gpu_available = torch.cuda.is_available()
        device = "cuda" if gpu_available else "cpu"

        logger.info(f"Device: {device}")
        if gpu_available:
            logger.info(f"GPU: {torch.cuda.get_device_name(0)}")
            logger.info(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

        # Initialize pipeline
        models_dir = os.environ.get("MODELS_DIR", "./models")
        use_scene_planner = os.environ.get("USE_SCENE_PLANNER", "true").lower() == "true"

        logger.info(f"Models directory: {models_dir}")
        logger.info(f"Scene planner: {'enabled' if use_scene_planner else 'disabled'}")

        pipeline = CompleteInflightSelfiePipeline(
            device=device,
            models_dir=models_dir,
            use_scene_planner=use_scene_planner,
            enable_cpu_offload=not gpu_available or os.environ.get("CPU_OFFLOAD", "true").lower() == "true",
        )

        logger.info("✅ Pipeline initialized successfully!")

    except Exception as e:
        logger.error(f"❌ Failed to initialize pipeline: {e}")
        raise


@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup on server shutdown."""
    global pipeline

    logger.info("🛑 Shutting down...")

    if pipeline is not None:
        # Clear GPU memory
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    logger.info("✅ Shutdown complete")


# Helper functions
def save_uploaded_file(upload_file: UploadFile) -> str:
    """Save uploaded file to temporary location."""
    try:
        suffix = Path(upload_file.filename).suffix
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            content = upload_file.file.read()
            tmp.write(content)
            return tmp.name
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Failed to save uploaded file: {e}")


def image_to_base64(image: Image.Image, format: str = "PNG") -> str:
    """Convert PIL Image to base64 string."""
    buffer = io.BytesIO()
    image.save(buffer, format=format)
    img_bytes = buffer.getvalue()
    img_base64 = base64.b64encode(img_bytes).decode()
    return f"data:image/{format.lower()};base64,{img_base64}"


def cleanup_temp_files(file_paths: List[str]):
    """Delete temporary files."""
    for path in file_paths:
        try:
            if os.path.exists(path):
                os.unlink(path)
        except Exception as e:
            logger.warning(f"Failed to delete temp file {path}: {e}")


# API Endpoints
@app.get("/")
async def root():
    """Root endpoint with API information."""
    return {
        "name": "Inflight Selfie Generator API",
        "version": "1.0.0",
        "status": "running",
        "endpoints": {
            "health": "/health",
            "generate": "/generate (POST)",
            "generate_stream": "/generate/stream (POST)",
            "docs": "/docs",
        }
    }


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    global pipeline

    gpu_info = {}
    if torch.cuda.is_available():
        gpu_info = {
            "available": True,
            "name": torch.cuda.get_device_name(0),
            "memory_allocated_gb": torch.cuda.memory_allocated(0) / 1e9,
            "memory_reserved_gb": torch.cuda.memory_reserved(0) / 1e9,
        }
    else:
        gpu_info = {"available": False}

    return {
        "status": "healthy",
        "pipeline_loaded": pipeline is not None,
        "gpu": gpu_info,
    }


@app.post("/generate", response_model=GenerationResponse)
async def generate_selfie(
    prompt: str = Form(..., description="Scene description (e.g., 'sunset selfie flying to Dubai')"),
    person1_images: List[UploadFile] = File(..., description="1-5 face images of person 1"),
    person2_images: List[UploadFile] = File(..., description="1-5 face images of person 2"),
    num_inference_steps: int = Form(30, ge=20, le=50, description="Number of diffusion steps"),
    seed: Optional[int] = Form(None, description="Random seed for reproducibility"),
    ip_adapter_scale: Optional[float] = Form(None, ge=0.0, le=1.0, description="Identity preservation (0-1)"),
    guidance_scale: Optional[float] = Form(None, ge=1.0, le=20.0, description="Prompt adherence (1-20)"),
):
    """
    Generate inflight selfie from uploaded face photos.

    **Parameters:**
    - **prompt**: Scene description (e.g., "Two friends taking sunset selfie flying to Dubai")
    - **person1_images**: Upload 1-5 clear face photos of person 1 (more = better consistency)
    - **person2_images**: Upload 1-5 clear face photos of person 2
    - **num_inference_steps**: Higher = better quality but slower (20-50, default: 30)
    - **seed**: Set a number for reproducible results (optional)
    - **ip_adapter_scale**: How much to preserve face identity 0-1 (optional, auto-optimized)
    - **guidance_scale**: How much to follow prompt 1-20 (optional, auto-optimized)

    **Returns:**
    - Base64 encoded generated image
    - Parameters used for generation
    """
    global pipeline

    if pipeline is None:
        raise HTTPException(status_code=503, detail="Pipeline not initialized")

    temp_files = []

    try:
        # Validate inputs
        if len(person1_images) == 0 or len(person2_images) == 0:
            raise HTTPException(status_code=400, detail="Must upload at least one image per person")

        if len(person1_images) > 5 or len(person2_images) > 5:
            raise HTTPException(status_code=400, detail="Maximum 5 images per person")

        logger.info(f"Received generation request: '{prompt}'")
        logger.info(f"Person 1: {len(person1_images)} images, Person 2: {len(person2_images)} images")

        # Save uploaded files
        logger.info("Saving uploaded files...")
        person1_paths = [save_uploaded_file(f) for f in person1_images]
        person2_paths = [save_uploaded_file(f) for f in person2_images]
        temp_files.extend(person1_paths + person2_paths)

        # Prepare override parameters
        override_params = {}
        if ip_adapter_scale is not None:
            override_params["ip_adapter_scale"] = ip_adapter_scale
        if guidance_scale is not None:
            override_params["guidance_scale"] = guidance_scale

        # Generate
        logger.info("Starting generation...")
        result_image = pipeline.generate(
            user_prompt=prompt,
            person1_images=person1_paths,
            person2_images=person2_paths,
            num_inference_steps=num_inference_steps,
            seed=seed,
            override_params=override_params if override_params else None,
        )

        # Convert to base64
        logger.info("Encoding result...")
        img_base64 = image_to_base64(result_image)

        # Get actual parameters used (from scene planner or defaults)
        actual_params = {
            "num_inference_steps": num_inference_steps,
            "seed": seed,
            "ip_adapter_scale": override_params.get("ip_adapter_scale", "auto"),
            "guidance_scale": override_params.get("guidance_scale", "auto"),
        }

        logger.info("✅ Generation successful!")

        return GenerationResponse(
            success=True,
            image=img_base64,
            prompt_used=prompt,
            parameters=actual_params,
        )

    except ValueError as e:
        # Face detection or validation errors
        logger.error(f"Validation error: {e}")
        raise HTTPException(status_code=400, detail=str(e))

    except Exception as e:
        # Other errors
        logger.error(f"Generation error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Generation failed: {str(e)}")

    finally:
        # Cleanup temporary files
        cleanup_temp_files(temp_files)


@app.post("/generate/stream")
async def generate_selfie_stream(
    prompt: str = Form(...),
    person1_images: List[UploadFile] = File(...),
    person2_images: List[UploadFile] = File(...),
    num_inference_steps: int = Form(30, ge=20, le=50),
    seed: Optional[int] = Form(None),
):
    """
    Generate inflight selfie and return as streaming image response.

    Same parameters as /generate but returns PNG directly instead of JSON.
    Useful for direct image display.
    """
    global pipeline

    if pipeline is None:
        raise HTTPException(status_code=503, detail="Pipeline not initialized")

    temp_files = []

    try:
        # Save uploaded files
        person1_paths = [save_uploaded_file(f) for f in person1_images]
        person2_paths = [save_uploaded_file(f) for f in person2_images]
        temp_files.extend(person1_paths + person2_paths)

        # Generate
        result_image = pipeline.generate(
            user_prompt=prompt,
            person1_images=person1_paths,
            person2_images=person2_paths,
            num_inference_steps=num_inference_steps,
            seed=seed,
        )

        # Convert to bytes
        img_bytes = io.BytesIO()
        result_image.save(img_bytes, format="PNG")
        img_bytes.seek(0)

        return StreamingResponse(img_bytes, media_type="image/png")

    except Exception as e:
        logger.error(f"Generation error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

    finally:
        cleanup_temp_files(temp_files)


@app.get("/examples")
async def get_examples():
    """Get example prompts for inspiration."""
    return {
        "examples": [
            {
                "title": "Sunset Window Seat",
                "prompt": "Two friends taking a sunset selfie flying to Dubai, window seat, golden hour lighting",
                "scene_type": "sunset_window"
            },
            {
                "title": "Business Class Celebration",
                "prompt": "Business class celebration selfie with champagne, luxury cabin",
                "scene_type": "business_celebration"
            },
            {
                "title": "Night City Lights",
                "prompt": "Night flight selfie with city lights below, dim cabin lighting",
                "scene_type": "night_city"
            },
            {
                "title": "Morning Clouds",
                "prompt": "Morning flight selfie over fluffy white clouds, bright sunlight",
                "scene_type": "morning_clouds"
            },
            {
                "title": "Tropical Destination",
                "prompt": "Selfie flying over tropical ocean and islands, vacation excitement",
                "scene_type": "tropical"
            },
            {
                "title": "First Class Luxury",
                "prompt": "First class suite selfie, spacious luxury cabin with premium amenities",
                "scene_type": "first_class"
            },
            {
                "title": "Takeoff Excitement",
                "prompt": "Excited selfie during takeoff with runway visible through window",
                "scene_type": "takeoff"
            },
            {
                "title": "Landing Celebration",
                "prompt": "Happy landing selfie with destination airport visible, arrival excitement",
                "scene_type": "landing"
            },
        ]
    }


# Error handlers
@app.exception_handler(HTTPException)
async def http_exception_handler(request, exc):
    """Custom HTTP exception handler."""
    return JSONResponse(
        status_code=exc.status_code,
        content=ErrorResponse(
            success=False,
            error=exc.detail,
        ).dict()
    )


@app.exception_handler(Exception)
async def general_exception_handler(request, exc):
    """General exception handler."""
    logger.error(f"Unhandled exception: {exc}", exc_info=True)
    return JSONResponse(
        status_code=500,
        content=ErrorResponse(
            success=False,
            error="Internal server error",
            detail=str(exc) if os.environ.get("DEBUG", "false").lower() == "true" else None,
        ).dict()
    )


if __name__ == "__main__":
    import uvicorn

    # Run server
    uvicorn.run(
        "server:app",
        host=os.environ.get("HOST", "0.0.0.0"),
        port=int(os.environ.get("PORT", 8000)),
        reload=os.environ.get("RELOAD", "false").lower() == "true",
        log_level=os.environ.get("LOG_LEVEL", "info"),
    )
