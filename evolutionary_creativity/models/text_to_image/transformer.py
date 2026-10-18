"""Diffusion transformer (DiT, flow matching) models of the model comparison."""
import diffusers
import torch
from PIL import Image

from evolutionary_creativity.models.text_to_image.base import GenerativeModel, ModelNoiseConfig, timed_call


class DiffusersTransformerModel(GenerativeModel):
    """Shared wrapper for transformer-based Diffusers pipelines in bfloat16 at 1024 x 1024.

    Subclasses set the class attributes below. The pipeline class is looked up by name, so a model only
    requires its pipeline to exist in the installed ``diffusers`` version when it is actually used.

    Attributes:
        pipeline_name: Name of the Diffusers pipeline class.
        default_model: Hugging Face model id.
        default_num_inference_steps: Number of denoising steps.
        default_guidance_scale: Guidance scale passed to the pipeline; ``None`` keeps the pipeline default.
        load_kwargs: Extra keyword arguments for ``from_pretrained``.
        num_channels_latents: Number of channels of the (patch-packed) latents the GA works on.
        vae_scale_factor: Factor between image size and latent grid.
        noise_dtype: Data type of the noise; ``None`` uses the pipeline dtype.
    """

    pipeline_name: str = ""
    default_model: str = ""
    default_num_inference_steps: int = 50
    default_guidance_scale: float | None = None
    load_kwargs: dict = {}
    num_channels_latents: int = 16
    vae_scale_factor: int = 8
    noise_dtype: torch.dtype | None = None

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
        self.num_inference_steps = (num_inference_steps if num_inference_steps is not None
                                    else self.default_num_inference_steps)
        self.guidance_scale = guidance_scale if guidance_scale is not None else self.default_guidance_scale
        pipeline_class = getattr(diffusers, self.pipeline_name)
        self.model = pipeline_class.from_pretrained(model or self.default_model, cache_dir=cache_dir,
                                                    dtype=torch.bfloat16, use_safetensors=True, **self.load_kwargs)
        self.model.set_progress_bar_config(leave=False)
        self.model.set_progress_bar_config(disable=True)
        self.model.to(device=device)

    def _prepare_latents(self, latents: torch.Tensor) -> torch.Tensor:
        """Convert the GA latents ``[B, C, H, W]`` into the format the pipeline expects.

        Args:
            latents: Stacked initial noise.

        Returns:
            Latents as passed to the pipeline (unchanged by default).
        """
        return latents

    def generate_batch(self, noises: list[torch.Tensor], prompt: str) -> list[Image.Image]:
        """Generate one image per noise tensor.

        Args:
            noises: Initial noise tensors matching :meth:`noise_config`.
            prompt: Text prompt shared by all images.

        Returns:
            Generated 1024 x 1024 images.
        """
        latents = self._prepare_latents(torch.cat(noises, dim=0))
        kwargs = {} if self.guidance_scale is None else {"guidance_scale": self.guidance_scale}

        def generate():
            """Run the pipeline on the prepared latents."""
            return self.model(prompt=[prompt] * len(noises), latents=latents, output_type="pil",
                              num_inference_steps=self.num_inference_steps, **kwargs).images

        return timed_call(type(self).__name__, len(noises), generate)

    def noise_config(self) -> ModelNoiseConfig:
        """Return the latent layout given by the class attributes."""
        return ModelNoiseConfig(num_channels_latents=self.num_channels_latents, vae_scale_factor=self.vae_scale_factor,
                                default_height=1024, default_width=1024, dtype=self.noise_dtype or self.model.dtype)


class ZImage(DiffusersTransformerModel):
    """Z-Image (Tongyi-MAI), 50 steps with the pipeline's default guidance."""
    pipeline_name = "ZImagePipeline"
    default_model = "Tongyi-MAI/Z-Image"
    default_num_inference_steps = 50
    load_kwargs = {"low_cpu_mem_usage": False}


class Flux2Klein4B(DiffusersTransformerModel):
    """FLUX.2 [klein] 4B, distilled to 4 steps.

    FLUX.2 packs the 32-channel VAE latents into 2 x 2 patches: ``[B, 128, H/16, W/16]``.
    """
    pipeline_name = "Flux2KleinPipeline"
    default_model = "black-forest-labs/FLUX.2-klein-4B"
    default_num_inference_steps = 4
    default_guidance_scale = 1.0
    num_channels_latents = 128
    vae_scale_factor = 16


class Flux2Klein9BKV(DiffusersTransformerModel):
    """FLUX.2 [klein] 9B with KV cache, distilled to 4 steps, pipeline default guidance."""
    pipeline_name = "Flux2KleinKVPipeline"
    default_model = "black-forest-labs/FLUX.2-klein-9b-kv"
    default_num_inference_steps = 4
    num_channels_latents = 128
    vae_scale_factor = 16


class Flux2Dev(DiffusersTransformerModel):
    """FLUX.2 [dev] with 4-bit (bitsandbytes) weights, 50 steps, guidance 4.

    The noise is fixed to bfloat16 because the pipeline dtype reports ``uint8`` for 4-bit weights.
    """
    pipeline_name = "Flux2Pipeline"
    default_model = "diffusers/FLUX.2-dev-bnb-4bit"
    default_num_inference_steps = 50
    default_guidance_scale = 4.0
    num_channels_latents = 128
    vae_scale_factor = 16
    noise_dtype = torch.bfloat16


class Krea2Turbo(DiffusersTransformerModel):
    """Krea 2 Turbo, 8 steps without guidance."""
    pipeline_name = "Krea2Pipeline"
    default_model = "krea/Krea-2-Turbo"
    default_num_inference_steps = 8
    default_guidance_scale = 0.0

    def _prepare_latents(self, latents: torch.Tensor) -> torch.Tensor:
        """Pack ``[B, C, H, W]`` into ``[B, H/p * W/p, C * p * p]``; the pipeline does not pack given latents.

        Args:
            latents: Stacked initial noise.

        Returns:
            Packed latents.
        """
        b, c, h, w = latents.shape
        return self.model._pack_latents(latents, b, c, h, w)
