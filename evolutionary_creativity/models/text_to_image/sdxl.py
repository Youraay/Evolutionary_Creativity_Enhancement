"""Stable Diffusion XL and SDXL-Turbo."""
import torch
from diffusers import AutoPipelineForText2Image, StableDiffusionXLPipeline
from PIL import Image

from evolutionary_creativity.models.text_to_image.base import GenerativeModel, ModelNoiseConfig, timed_call


class StableDiffusionXLModel(GenerativeModel):
    """SDXL base 1.0 in bfloat16, 1024 x 1024 (latents ``[1, 4, 128, 128]``).

    bfloat16 instead of float16 avoids the float32 upcast of the SDXL VAE (``force_upcast``).
    """

    pipeline_class = AutoPipelineForText2Image
    torch_dtype = torch.bfloat16

    def __init__(self, device: str, cache_dir: str, num_inference_steps: int = 50, guidance_scale: float = 7.0,
                 model: str = "stabilityai/stable-diffusion-xl-base-1.0",
                 torch_dtype: torch.dtype | None = None) -> None:
        """
        Args:
            device: Device of the pipeline.
            cache_dir: Hugging Face cache directory.
            num_inference_steps: Number of denoising steps (50 for the experiments with population 100, 30 for
                those with population 60).
            guidance_scale: Classifier-free guidance scale.
            model: Hugging Face model id.
            torch_dtype: Precision of the pipeline; by default :attr:`torch_dtype` of the class. The experiments
                with population 100 use float16.
        """
        self.num_inference_steps = num_inference_steps
        self.guidance_scale = guidance_scale
        self.model = self.pipeline_class.from_pretrained(model, cache_dir=cache_dir,
                                                         torch_dtype=torch_dtype or self.torch_dtype,
                                                         use_safetensors=True, variant="fp16")
        self.model.set_progress_bar_config(leave=False)
        self.model.set_progress_bar_config(disable=True)
        self.model.to(device=device)

    def generate_batch(self, noises: list[torch.Tensor], prompt: str) -> list[Image.Image]:
        """Generate one image per noise tensor.

        Args:
            noises: Initial noise tensors of shape ``[1, 4, 128, 128]``.
            prompt: Text prompt shared by all images.

        Returns:
            Generated 1024 x 1024 images.
        """
        latents = torch.cat(noises, dim=0)

        def generate():
            """Run the pipeline on the stacked latents."""
            with torch.inference_mode():
                return self.model(prompt=[prompt] * len(noises), latents=latents, output_type="pil",
                                  num_inference_steps=self.num_inference_steps,
                                  guidance_scale=self.guidance_scale).images

        return timed_call("SDXL", len(noises), generate)

    def noise_config(self) -> ModelNoiseConfig:
        """Return the SDXL latent layout (4 channels, factor 8, 1024 x 1024, pipeline dtype)."""
        return ModelNoiseConfig(num_channels_latents=4, vae_scale_factor=8, default_height=1024, default_width=1024,
                                dtype=self.model.dtype)


class StableDiffusionXLTurboModel(StableDiffusionXLModel):
    """SDXL-Turbo in float16: one step without classifier-free guidance.

    The latents use the same 1024 x 1024 layout as SDXL (``[1, 4, 128, 128]``).
    """

    pipeline_class = StableDiffusionXLPipeline
    torch_dtype = torch.float16

    def __init__(self, device: str, cache_dir: str, num_inference_steps: int = 1, guidance_scale: float = 0.0,
                 model: str = "stabilityai/sdxl-turbo") -> None:
        """
        Args:
            device: Device of the pipeline.
            cache_dir: Hugging Face cache directory.
            num_inference_steps: Number of denoising steps.
            guidance_scale: Classifier-free guidance scale (0 disables it).
            model: Hugging Face model id.
        """
        super().__init__(device, cache_dir, num_inference_steps, guidance_scale, model)
