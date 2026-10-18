"""Common interface of the text-to-image models."""
import logging
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass

import torch
from PIL import Image

logger = logging.getLogger(__name__)


@dataclass
class ModelNoiseConfig:
    """Layout of the initial noise a text-to-image model expects.

    Attributes:
        num_channels_latents: Number of latent channels ``C``.
        vae_scale_factor: Spatial down-sampling factor ``f`` between image and latent.
        default_height: Image height in pixels; the latent height is ``default_height // f``.
        default_width: Image width in pixels; the latent width is ``default_width // f``.
        dtype: Data type of the noise.
    """

    num_channels_latents: int
    vae_scale_factor: int
    default_height: int
    default_width: int
    dtype: torch.dtype = torch.float16


class GenerativeModel(ABC):
    """Text-to-image model that turns given initial noise into images."""

    @abstractmethod
    def generate_batch(self, noises: list[torch.Tensor], prompt: str) -> list[Image.Image]:
        """Generate one image per noise tensor.

        Args:
            noises: Initial noise tensors of shape ``[1, C, H, W]`` (see :meth:`noise_config`).
            prompt: Text prompt shared by all images.

        Returns:
            Generated images in the order of ``noises``.
        """

    @abstractmethod
    def noise_config(self) -> ModelNoiseConfig:
        """Return the noise layout this model expects."""


def timed_call(model_name: str, batch_size: int, generate) -> list[Image.Image]:
    """Run ``generate()`` and log the wall time per image.

    Args:
        model_name: Name used in the log line.
        batch_size: Number of images produced by ``generate``.
        generate: Zero-argument callable returning the images.

    Returns:
        The images returned by ``generate``.
    """
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    start = time.time()
    images = generate()
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    seconds = time.time() - start
    logger.info("[TIMING] %s batch=%d took %.3fs (%.3fs/image)", model_name, batch_size, seconds, seconds / batch_size)
    return images
