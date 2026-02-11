"""
Inflight Selfie Generator - Production Pipeline
Combines TinyLlama scene planner with IP-Adapter-FaceID image generation
"""

import os
import json
import torch
import cv2
import numpy as np
from PIL import Image
from typing import List, Union, Optional, Dict
from pathlib import Path

# Diffusion imports
from diffusers import StableDiffusionXLPipeline, DDIMScheduler

# Face analysis
from insightface.app import FaceAnalysis

# LLM imports
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel


class InflightSelfiePipeline:
    """
    IP-Adapter-FaceID pipeline for generating inflight selfies.

    This handles face embedding extraction and image generation.
    """

    def __init__(
        self,
        device: str = "cuda",
        models_dir: str = "./models",
        enable_cpu_offload: bool = True,
    ):
        """
        Initialize the pipeline.

        Args:
            device: Device to run on ('cuda' or 'cpu')
            models_dir: Directory containing models
            enable_cpu_offload: Enable CPU offloading for memory efficiency
        """
        self.device = device
        self.models_dir = Path(models_dir)
        self.enable_cpu_offload = enable_cpu_offload

        self.face_analyzer = None
        self.pipe = None

        print("🚀 Initializing Inflight Selfie Pipeline...")
        self._setup_face_analyzer()
        self._setup_diffusion_pipeline()
        print("✅ Pipeline ready!")

    def _setup_face_analyzer(self):
        """Initialize InsightFace for face embedding extraction."""
        print("  Loading face analyzer...")

        insightface_path = self.models_dir / "insightface"

        self.face_analyzer = FaceAnalysis(
            name='antelopev2',
            root=str(insightface_path),
            providers=['CUDAExecutionProvider', 'CPUExecutionProvider']
        )
        self.face_analyzer.prepare(ctx_id=0 if self.device == "cuda" else -1, det_size=(640, 640))
        print("    ✓ Face analyzer ready")

    def _setup_diffusion_pipeline(self):
        """Initialize SDXL with IP-Adapter-FaceID."""
        print("  Loading SDXL pipeline...")

        # Load base SDXL
        self.pipe = StableDiffusionXLPipeline.from_pretrained(
            "stabilityai/stable-diffusion-xl-base-1.0",
            torch_dtype=torch.float16,
            variant="fp16",
        )

        if self.device == "cuda":
            self.pipe = self.pipe.to(self.device)

        # Use efficient scheduler
        self.pipe.scheduler = DDIMScheduler.from_config(self.pipe.scheduler.config)

        # Load IP-Adapter-FaceID
        print("    Loading IP-Adapter-FaceID...")
        self.pipe.load_ip_adapter(
            "h94/IP-Adapter-FaceID",
            subfolder=None,
            weight_name="ip-adapter-faceid_sdxl.bin",
        )

        # Enable memory optimizations
        if self.enable_cpu_offload:
            self.pipe.enable_model_cpu_offload()

        self.pipe.enable_vae_slicing()

        print("    ✓ SDXL + IP-Adapter-FaceID ready")

    def extract_face_embedding(self, image: Union[str, np.ndarray, Image.Image]) -> np.ndarray:
        """
        Extract face embedding from a single image.

        Args:
            image: Image as file path, numpy array, or PIL Image

        Returns:
            Face embedding as numpy array (512-dimensional)

        Raises:
            ValueError: If no face is detected
        """
        # Convert to numpy array
        if isinstance(image, str):
            img = cv2.imread(image)
            if img is None:
                raise ValueError(f"Failed to load image: {image}")
        elif isinstance(image, Image.Image):
            img = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
        else:
            img = image

        # Detect faces
        faces = self.face_analyzer.get(img)

        if not faces:
            raise ValueError("No face detected in image")

        # Return embedding of largest face
        face = max(faces, key=lambda x: (x.bbox[2] - x.bbox[0]) * (x.bbox[3] - x.bbox[1]))
        return face.embedding

    def extract_face_embeddings_multi(
        self,
        images: List[Union[str, np.ndarray, Image.Image]]
    ) -> np.ndarray:
        """
        Extract and average embeddings from multiple images of the same person.

        Using multiple images improves identity consistency.

        Args:
            images: List of images (paths, arrays, or PIL Images)

        Returns:
            Averaged face embedding

        Raises:
            ValueError: If no faces detected in any image
        """
        embeddings = []

        for i, img in enumerate(images):
            try:
                emb = self.extract_face_embedding(img)
                embeddings.append(emb)
            except ValueError as e:
                print(f"    ⚠️ Warning: Image {i+1}: {e}")
                continue

        if not embeddings:
            raise ValueError("No faces detected in any images")

        # Average embeddings
        avg_embedding = np.mean(embeddings, axis=0)
        print(f"    ✓ Extracted {len(embeddings)} face embeddings")
        return avg_embedding

    def generate_selfie(
        self,
        person1_images: List[Union[str, np.ndarray, Image.Image]],
        person2_images: List[Union[str, np.ndarray, Image.Image]],
        prompt: str,
        negative_prompt: Optional[str] = None,
        num_inference_steps: int = 30,
        guidance_scale: float = 7.5,
        ip_adapter_scale: float = 0.65,
        seed: Optional[int] = None,
        height: int = 1024,
        width: int = 1024,
    ) -> Image.Image:
        """
        Generate inflight selfie with two people.

        Args:
            person1_images: List of images for person 1
            person2_images: List of images for person 2
            prompt: Scene description
            negative_prompt: What to avoid (optional)
            num_inference_steps: Number of diffusion steps (20-50)
            guidance_scale: Classifier-free guidance scale (5-10)
            ip_adapter_scale: Identity preservation strength (0.5-0.8)
            seed: Random seed for reproducibility
            height: Output height in pixels
            width: Output width in pixels

        Returns:
            Generated PIL Image
        """

        # Extract face embeddings
        print("  Extracting face embeddings...")
        emb1 = self.extract_face_embeddings_multi(person1_images)
        emb2 = self.extract_face_embeddings_multi(person2_images)

        # Combine embeddings (weighted average)
        # Note: This is a simplified approach. For better multi-person results:
        # 1. Generate separate images and composite
        # 2. Use IP-Adapter-FaceID-PlusV2 with multi-face support
        # 3. Use ControlNet for pose guidance
        combined_emb = (emb1 + emb2) / 2
        face_emb_tensor = torch.tensor(combined_emb, dtype=torch.float16).unsqueeze(0).to(self.device)

        # Default negative prompt
        if negative_prompt is None:
            negative_prompt = (
                "ugly, blurry, low quality, distorted face, bad anatomy, "
                "deformed, disfigured, watermark, text, oversaturated, "
                "extra limbs, missing limbs, floating limbs, mutation, "
                "duplicate faces, bad eyes, asymmetric eyes, bad proportions"
            )

        # Set IP-Adapter scale
        self.pipe.set_ip_adapter_scale(ip_adapter_scale)

        # Set seed if provided
        generator = None
        if seed is not None:
            generator = torch.Generator(device=self.device).manual_seed(seed)

        # Generate
        print(f"  Generating {width}x{height} image...")
        result = self.pipe(
            prompt=prompt,
            negative_prompt=negative_prompt,
            ip_adapter_image_embeds=[face_emb_tensor],
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
            generator=generator,
            height=height,
            width=width,
        )

        print("  ✅ Generation complete!")
        return result.images[0]


class ScenePlanner:
    """
    TinyLlama-based scene planner for generating optimal generation parameters.
    """

    def __init__(
        self,
        device: str = "cuda",
        models_dir: str = "./models",
        use_finetuned: bool = True,
    ):
        """
        Initialize the scene planner.

        Args:
            device: Device to run on
            models_dir: Directory containing models
            use_finetuned: Whether to use fine-tuned LoRA weights
        """
        self.device = device
        self.models_dir = Path(models_dir)
        self.use_finetuned = use_finetuned

        self.model = None
        self.tokenizer = None

        print("🧠 Loading scene planner...")
        self._setup_model()
        print("✅ Scene planner ready!")

    def _setup_model(self):
        """Load TinyLlama with optional LoRA weights."""

        lora_path = self.models_dir / "scene_planner_lora"

        if self.use_finetuned and lora_path.exists():
            print(f"  Loading fine-tuned model from {lora_path}...")

            # Load base model
            base_model = AutoModelForCausalLM.from_pretrained(
                "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
                torch_dtype=torch.float16,
                device_map="auto"
            )

            # Load LoRA weights
            self.model = PeftModel.from_pretrained(base_model, str(lora_path))
            self.tokenizer = AutoTokenizer.from_pretrained(str(lora_path))

            print("    ✓ Loaded fine-tuned scene planner")
        else:
            print("  Loading base TinyLlama model...")

            self.model = AutoModelForCausalLM.from_pretrained(
                "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
                torch_dtype=torch.float16,
                device_map="auto"
            )
            self.tokenizer = AutoTokenizer.from_pretrained("TinyLlama/TinyLlama-1.1B-Chat-v1.0")

            if self.use_finetuned:
                print("    ⚠️ Fine-tuned weights not found, using base model")
            else:
                print("    ✓ Loaded base scene planner")

    def plan_scene(self, user_prompt: str, temperature: float = 0.7) -> Dict:
        """
        Generate scene parameters from user prompt.

        Args:
            user_prompt: Natural language scene description
            temperature: Sampling temperature (0.1-1.0)

        Returns:
            Dictionary with generation parameters
        """

        full_prompt = f"""<|system|>
You are an inflight selfie scene planner. Given a user's description, output a JSON configuration with optimal parameters for generating a realistic inflight selfie.
<|user|>
{user_prompt}
<|assistant|>
"""

        inputs = self.tokenizer(full_prompt, return_tensors="pt").to(self.model.device)

        outputs = self.model.generate(
            **inputs,
            max_new_tokens=400,
            temperature=temperature,
            do_sample=True,
            pad_token_id=self.tokenizer.eos_token_id,
            eos_token_id=self.tokenizer.eos_token_id,
        )

        response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)

        # Extract JSON from response
        try:
            json_start = response.rfind("{")
            json_end = response.rfind("}") + 1

            if json_start >= 0 and json_end > json_start:
                params = json.loads(response[json_start:json_end])

                # Validate and set defaults
                params.setdefault("prompt", f"two people taking selfie in airplane, {user_prompt}")
                params.setdefault("negative_prompt", "ugly, blurry, distorted")
                params.setdefault("ip_adapter_scale", 0.65)
                params.setdefault("guidance_scale", 7.5)

                return params
        except json.JSONDecodeError:
            pass

        # Fallback defaults
        return {
            "prompt": f"two people taking selfie in airplane, {user_prompt}, high quality photo, realistic, detailed faces",
            "negative_prompt": "ugly, blurry, distorted, bad anatomy, deformed",
            "ip_adapter_scale": 0.65,
            "guidance_scale": 7.5,
            "scene_type": "custom",
        }


class CompleteInflightSelfiePipeline:
    """
    Complete pipeline combining scene planning and image generation.

    Usage:
        pipeline = CompleteInflightSelfiePipeline()
        result = pipeline.generate(
            user_prompt="sunset selfie flying to Dubai",
            person1_images=["person1_face1.jpg", "person1_face2.jpg"],
            person2_images=["person2_face1.jpg", "person2_face2.jpg"],
        )
        result.save("output.png")
    """

    def __init__(
        self,
        device: str = "cuda",
        models_dir: str = "./models",
        use_scene_planner: bool = True,
        enable_cpu_offload: bool = True,
    ):
        """
        Initialize the complete pipeline.

        Args:
            device: Device to run on ('cuda' or 'cpu')
            models_dir: Directory containing models
            use_scene_planner: Whether to use TinyLlama scene planner
            enable_cpu_offload: Enable CPU offloading for memory efficiency
        """
        self.device = device
        self.models_dir = models_dir
        self.use_scene_planner = use_scene_planner

        print("🚀 Initializing Complete Inflight Selfie Pipeline...")
        print("="*60)

        # Initialize image generation pipeline
        self.image_pipeline = InflightSelfiePipeline(
            device=device,
            models_dir=models_dir,
            enable_cpu_offload=enable_cpu_offload,
        )

        # Initialize scene planner (optional)
        if self.use_scene_planner:
            self.scene_planner = ScenePlanner(
                device=device,
                models_dir=models_dir,
                use_finetuned=True,
            )
        else:
            self.scene_planner = None

        print("="*60)
        print("✅ Complete pipeline ready!")
        print()

    def generate(
        self,
        user_prompt: str,
        person1_images: List[Union[str, np.ndarray, Image.Image]],
        person2_images: List[Union[str, np.ndarray, Image.Image]],
        num_inference_steps: int = 30,
        seed: Optional[int] = None,
        override_params: Optional[Dict] = None,
    ) -> Image.Image:
        """
        Generate inflight selfie from natural language prompt.

        Args:
            user_prompt: Natural language scene description
            person1_images: List of images for person 1 (1-5 recommended)
            person2_images: List of images for person 2 (1-5 recommended)
            num_inference_steps: Number of diffusion steps (20-50)
            seed: Random seed for reproducibility
            override_params: Optional dict to override scene planner parameters

        Returns:
            Generated PIL Image
        """

        print("\n" + "="*60)
        print("🎬 GENERATING INFLIGHT SELFIE")
        print("="*60)

        # Step 1: Plan the scene (if enabled)
        if self.use_scene_planner and self.scene_planner is not None:
            print("\n📋 Step 1: Planning scene...")
            scene_params = self.scene_planner.plan_scene(user_prompt)

            print(f"  Scene type: {scene_params.get('scene_type', 'custom')}")
            print(f"  Prompt: {scene_params['prompt'][:80]}...")
            print(f"  IP Scale: {scene_params.get('ip_adapter_scale', 0.65)}")
            print(f"  CFG Scale: {scene_params.get('guidance_scale', 7.5)}")
        else:
            print("\n📋 Step 1: Using default parameters...")
            scene_params = {
                "prompt": f"two people taking selfie in airplane, {user_prompt}, high quality photo, realistic, detailed faces",
                "negative_prompt": "ugly, blurry, distorted, bad anatomy",
                "ip_adapter_scale": 0.65,
                "guidance_scale": 7.5,
            }

        # Apply overrides
        if override_params:
            scene_params.update(override_params)

        # Step 2: Generate image
        print("\n🎨 Step 2: Generating image...")
        result_image = self.image_pipeline.generate_selfie(
            person1_images=person1_images,
            person2_images=person2_images,
            prompt=scene_params["prompt"],
            negative_prompt=scene_params.get("negative_prompt"),
            ip_adapter_scale=scene_params.get("ip_adapter_scale", 0.65),
            guidance_scale=scene_params.get("guidance_scale", 7.5),
            num_inference_steps=num_inference_steps,
            seed=seed,
        )

        print("\n" + "="*60)
        print("✅ GENERATION COMPLETE!")
        print("="*60 + "\n")

        return result_image


# Convenience function
def generate_inflight_selfie(
    user_prompt: str,
    person1_images: List[Union[str, np.ndarray, Image.Image]],
    person2_images: List[Union[str, np.ndarray, Image.Image]],
    output_path: Optional[str] = None,
    seed: Optional[int] = None,
) -> Image.Image:
    """
    Convenience function to generate inflight selfie.

    Args:
        user_prompt: Scene description (e.g., "sunset selfie flying to Dubai")
        person1_images: List of image paths/arrays for person 1
        person2_images: List of image paths/arrays for person 2
        output_path: Optional path to save result
        seed: Random seed for reproducibility

    Returns:
        Generated PIL Image
    """
    pipeline = CompleteInflightSelfiePipeline()
    result = pipeline.generate(
        user_prompt=user_prompt,
        person1_images=person1_images,
        person2_images=person2_images,
        seed=seed,
    )

    if output_path:
        result.save(output_path)
        print(f"💾 Saved to: {output_path}")

    return result


if __name__ == "__main__":
    print("Inflight Selfie Generator - Pipeline Module")
    print("Import this module to use the pipeline in your application")
