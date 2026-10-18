"""Creation of initial noise tensors and candidate ids."""
import random
from dataclasses import dataclass

import torch

from evolutionary_creativity.candidate import Candidate
from evolutionary_creativity.models.text_to_image.base import ModelNoiseConfig


@dataclass
class NoiseFactory:
    """Creates candidates with Gaussian initial noise and assigns consecutive ids.

    Attributes:
        device: Device of the ``torch.Generator`` and of the created noise.
        seed_range: Inclusive range from which the noise seed is drawn with Python's ``random``.
        id_count: Number of ids handed out so far.
    """

    device: str = "cuda"
    seed_range: tuple[int, int] = (1, 2**32 - 1)
    id_count: int = 0

    def create_id(self) -> str:
        """Return the next candidate id (``"n_0000"``, ``"n_0001"``, ...)."""
        candidate_id = f"n_{self.id_count:04}"
        self.id_count += 1
        return candidate_id

    def from_tensor(self, noise: torch.Tensor) -> Candidate:
        """Wrap an existing noise tensor in a new candidate with a fresh id.

        Args:
            noise: Initial noise of shape ``[1, C, H, W]``.

        Returns:
            Candidate holding ``noise``.
        """
        return Candidate(id=self.create_id(), initial_noise=noise)

    def create_population(self, config: ModelNoiseConfig, size: int, init_noise_sigma: float = 1.0) -> list[Candidate]:
        """Draw the initial population from a single seeded Gaussian.

        The seed comes from Python's ``random`` module, so the whole population is reproducible from the
        seed set at the start of a run.

        Args:
            config: Latent layout and dtype required by the text-to-image model.
            size: Number of candidates.
            init_noise_sigma: Factor applied to the standard normal noise.

        Returns:
            ``size`` candidates, each with noise of shape ``[1, C, H / f, W / f]`` (``f`` = VAE scale factor).
        """
        shape = (size, config.num_channels_latents,
                 config.default_height // config.vae_scale_factor,
                 config.default_width // config.vae_scale_factor)
        seed = random.randint(*self.seed_range)
        generator = torch.Generator(device=self.device).manual_seed(seed)
        noise = torch.randn(shape, generator=generator, device=self.device, dtype=config.dtype) * init_noise_sigma
        return [Candidate(id=self.create_id(), initial_noise=noise[i].unsqueeze(0).clone()) for i in range(size)]
