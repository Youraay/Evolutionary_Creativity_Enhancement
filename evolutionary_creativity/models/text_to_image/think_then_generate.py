"""Think-Then-Generate (T2G) diffusion transformer as generator."""
import torch
from diffusers import QwenImagePipeline
from diffusers.utils import logging as diffusers_logging
from PIL import Image

from evolutionary_creativity.models.text_to_image.base import GenerativeModel, ModelNoiseConfig


class ThinkThenGenerate(GenerativeModel):
    """T2G (arXiv 2601.10332): Qwen-Image pipeline with a jointly trained LLM encoder and a 20B DiT.

    The reasoning step (rewriting the prompt) happens beforehand
    (``experiments/think_then_generate/rewrite_prompts.py``); this class only generates from the prompt
    it receives, as ``inference.py`` of the T2G repository does (negative prompt ``" "``, true CFG).

    Prompt embeddings are computed once per prompt; afterwards the text encoder is moved to the CPU so
    that the DiT (41 GB), BLIP-2 and CLIP fit on one 80 GB GPU.
    """

    def __init__(self, device: str, cache_dir: str, num_inference_steps: int = 30, true_cfg_scale: float = 4.0,
                 negative_prompt: str = " ", height: int = 1024, width: int = 1024,
                 model: str = "SJTU-DENG-Lab/Think-Then-Generate-T2I") -> None:
        """
        Args:
            device: Device of the pipeline.
            cache_dir: Hugging Face cache directory.
            num_inference_steps: Number of denoising steps.
            true_cfg_scale: Scale of the true classifier-free guidance.
            negative_prompt: Negative prompt for the guidance.
            height: Image height in pixels.
            width: Image width in pixels.
            model: Hugging Face model id.
        """
        self.device = device
        self.num_inference_steps = num_inference_steps
        self.true_cfg_scale = true_cfg_scale
        self.negative_prompt = negative_prompt
        self.height, self.width = height, width
        self.model = QwenImagePipeline.from_pretrained(model, cache_dir=cache_dir, torch_dtype=torch.bfloat16)
        self.model.set_progress_bar_config(disable=True)
        self.model.to(device)
        diffusers_logging.set_verbosity_error()      # silences the missing prompt_embeds_mask hint for unpadded prompts
        self._embeds: dict[str, tuple[torch.Tensor, ...]] = {}

    def _prompt_embeds(self, prompt: str) -> tuple[torch.Tensor, ...]:
        """Encode prompt and negative prompt once and cache them.

        Args:
            prompt: Prompt to encode.

        Returns:
            ``(prompt_embeds, prompt_mask, negative_embeds, negative_mask)``.
        """
        if prompt not in self._embeds:
            encoder = self.model.text_encoder
            encoder.to(self.device)
            with torch.inference_mode():
                prompt_embeds, prompt_mask = self.model.encode_prompt(prompt=prompt, device=self.device)
                negative_embeds, negative_mask = self.model.encode_prompt(prompt=self.negative_prompt,
                                                                          device=self.device)
            self._embeds[prompt] = (prompt_embeds, prompt_mask, negative_embeds, negative_mask)
            encoder.to("cpu")
            torch.cuda.empty_cache()
        return self._embeds[prompt]

    def _pack(self, noise: torch.Tensor) -> torch.Tensor:
        """Reorder ``[B, 64, H/16, W/16]`` into the packed layout ``[B, H/16 * W/16, 64]`` of Qwen-Image.

        Qwen-Image expects packed 2 x 2 patches of its 16-channel latents. For i.i.d. noise this is only a
        permutation of the entries.

        Args:
            noise: Stacked initial noise.

        Returns:
            Packed latents in bfloat16 on the model device.
        """
        b, c = noise.shape[:2]
        return noise.permute(0, 2, 3, 1).reshape(b, -1, c).to(self.device, torch.bfloat16)

    def generate_batch(self, noises: list[torch.Tensor], prompt: str) -> list[Image.Image]:
        """Generate one image per noise tensor.

        Args:
            noises: Initial noise tensors of shape ``[1, 64, H/16, W/16]``.
            prompt: Text prompt shared by all images (usually the rewritten prompt).

        Returns:
            Generated images.
        """
        latents = self._pack(torch.cat(noises, dim=0))
        prompt_embeds, prompt_mask, negative_embeds, negative_mask = self._prompt_embeds(prompt)
        return self.model(prompt_embeds=prompt_embeds, prompt_embeds_mask=prompt_mask,
                          negative_prompt_embeds=negative_embeds, negative_prompt_embeds_mask=negative_mask,
                          num_images_per_prompt=latents.shape[0], latents=latents, height=self.height,
                          width=self.width, num_inference_steps=self.num_inference_steps,
                          true_cfg_scale=self.true_cfg_scale, output_type="pil").images

    def noise_config(self) -> ModelNoiseConfig:
        """Return the packed latent layout: 16 VAE channels x 2 x 2 patches = 64 channels on an ``H/16`` grid."""
        return ModelNoiseConfig(num_channels_latents=64, vae_scale_factor=16, default_height=self.height,
                                default_width=self.width, dtype=torch.bfloat16)
