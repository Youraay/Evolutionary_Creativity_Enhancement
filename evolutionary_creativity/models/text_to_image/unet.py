"""UNet-based diffusion models of the model comparison (SD 1.5, SD-Turbo and SDXL derivatives)."""
import torch
from diffusers import AutoPipelineForText2Image
from PIL import Image

from evolutionary_creativity.models.text_to_image.base import GenerativeModel, ModelNoiseConfig, timed_call


class DiffusersUNetModel(GenerativeModel):
    """Shared wrapper for UNet-based Diffusers pipelines.

    Latent shape and resolution are read from the loaded pipeline, so 512 px models (SD 1.5, SD-Turbo) and
    1024 px models (SDXL family) share this class. Scaling with ``scheduler.init_noise_sigma`` (e.g. EDM in
    Playground v2.5) is done by the pipeline itself. bfloat16 avoids the float32 upcast of SDXL VAEs.

    Subclasses set the class attributes below.

    Attributes:
        default_model: Hugging Face model id.
        default_num_inference_steps: Number of denoising steps.
        default_guidance_scale: Classifier-free guidance scale.
        variant: Weight variant to load (``None`` for the default weights).
        pipeline_kwargs: Extra keyword arguments for ``from_pretrained``.
    """

    default_model: str = ""
    default_num_inference_steps: int = 30
    default_guidance_scale: float = 7.0
    variant: str | None = "fp16"
    pipeline_kwargs: dict = {}

    def __init__(self, device: str, cache_dir: str, num_inference_steps: int | None = None,
                 guidance_scale: float | None = None, model: str | None = None) -> None:
        """
        Args:
            device: Device of the pipeline.
            cache_dir: Hugging Face cache directory.
            num_inference_steps: Overrides ``default_num_inference_steps``.
            guidance_scale: Overrides ``default_guidance_scale``.
            model: Overrides ``default_model``.
        """
        self.model_name = model or self.default_model
        self.num_inference_steps = (num_inference_steps if num_inference_steps is not None
                                    else self.default_num_inference_steps)
        self.guidance_scale = guidance_scale if guidance_scale is not None else self.default_guidance_scale

        self.model = AutoPipelineForText2Image.from_pretrained(self.model_name, cache_dir=cache_dir,
                                                               torch_dtype=torch.bfloat16, use_safetensors=True,
                                                               variant=self.variant, **self.pipeline_kwargs)
        self.model.set_progress_bar_config(leave=False)
        self.model.set_progress_bar_config(disable=True)
        self.model.to(device=device)

        self.height = self.model.unet.config.sample_size * self.model.vae_scale_factor
        self.width = self.height

    def generate_batch(self, noises: list[torch.Tensor], prompt: str) -> list[Image.Image]:
        """Generate one image per noise tensor at the native resolution of the model.

        Args:
            noises: Initial noise tensors matching :meth:`noise_config`.
            prompt: Text prompt shared by all images.

        Returns:
            Generated images.
        """
        latents = torch.cat(noises, dim=0)

        def generate():
            """Run the pipeline on the stacked latents."""
            with torch.inference_mode():
                return self.model(prompt=[prompt] * len(noises), latents=latents, height=self.height,
                                  width=self.width, output_type="pil", num_inference_steps=self.num_inference_steps,
                                  guidance_scale=self.guidance_scale).images

        return timed_call(type(self).__name__, len(noises), generate)

    def noise_config(self) -> ModelNoiseConfig:
        """Return the latent layout read from the UNet and VAE configuration."""
        return ModelNoiseConfig(num_channels_latents=self.model.unet.config.in_channels,
                                vae_scale_factor=self.model.vae_scale_factor, default_height=self.height,
                                default_width=self.width, dtype=self.model.dtype)


# The safety checker replaces flagged images with black images. In the genetic algorithm such outliers
# would distort the novelty fitness, so it is disabled for SD 1.5 and SD-Turbo.
_NO_SAFETY_CHECKER = {"safety_checker": None, "requires_safety_checker": False}


class StableDiffusion15Model(DiffusersUNetModel):
    """SD 1.5: 860M UNet, CLIP ViT-L text encoder, 512 x 512."""
    default_model = "stable-diffusion-v1-5/stable-diffusion-v1-5"
    default_num_inference_steps = 30
    default_guidance_scale = 7.5
    pipeline_kwargs = _NO_SAFETY_CHECKER


class SDTurboModel(DiffusersUNetModel):
    """SD-Turbo: adversarially distilled SD 2.1 (OpenCLIP ViT-H), 512 x 512, one step without guidance."""
    default_model = "stabilityai/sd-turbo"
    default_num_inference_steps = 1
    default_guidance_scale = 0.0
    pipeline_kwargs = _NO_SAFETY_CHECKER


class PlaygroundV25Model(DiffusersUNetModel):
    """Playground v2.5: SDXL architecture trained from scratch with an EDM scheduler, 1024 x 1024."""
    default_model = "playgroundai/playground-v2.5-1024px-aesthetic"
    default_num_inference_steps = 30
    default_guidance_scale = 3.0


class SSD1BModel(DiffusersUNetModel):
    """Segmind SSD-1B: SDXL UNet distilled to about 1.3B parameters, 1024 x 1024."""
    default_model = "segmind/SSD-1B"
    default_num_inference_steps = 30
    default_guidance_scale = 7.0


class SegmindVegaModel(DiffusersUNetModel):
    """Segmind Vega: SDXL UNet distilled to about 745M parameters, 1024 x 1024."""
    default_model = "segmind/Segmind-Vega"
    default_num_inference_steps = 30
    default_guidance_scale = 7.0


class RealVisXLModel(DiffusersUNetModel):
    """RealVisXL V5.0: photorealistic community fine-tune of SDXL, 1024 x 1024."""
    default_model = "SG161222/RealVisXL_V5.0"
    default_num_inference_steps = 30
    default_guidance_scale = 5.0


class JuggernautXLModel(DiffusersUNetModel):
    """Juggernaut XL v9: community fine-tune of SDXL, 1024 x 1024."""
    default_model = "RunDiffusion/Juggernaut-XL-v9"
    default_num_inference_steps = 30
    default_guidance_scale = 5.0
