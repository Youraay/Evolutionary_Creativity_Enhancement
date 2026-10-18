"""Mutation operators."""
from abc import ABC, abstractmethod

import torch


class MutationFunction(ABC):
    """Perturbs a single noise tensor."""

    @abstractmethod
    def mutate(self, noise: torch.Tensor) -> torch.Tensor:
        """Return a mutated copy of ``noise``.

        Args:
            noise: Noise tensor to mutate; it is not modified in place.

        Returns:
            Mutated tensor with the shape of ``noise``.
        """


class UniformGaussianMutator(MutationFunction):
    """Adds Gaussian noise to a random subset of the latent values.

    Exactly ``int(numel * mutation_rate)`` positions, chosen uniformly without replacement, receive
    ``N(0, mutation_strength^2)`` perturbations.
    """

    def __init__(self, mutation_rate: float, mutation_strength: float) -> None:
        """
        Args:
            mutation_rate: Fraction of the values that are perturbed.
            mutation_strength: Standard deviation of the added Gaussian noise.
        """
        self.mutation_rate = mutation_rate
        self.mutation_strength = mutation_strength

    def mutate(self, noise: torch.Tensor) -> torch.Tensor:
        """Perturb a random subset of the values of ``noise``.

        Args:
            noise: Noise tensor to mutate; it is not modified in place.

        Returns:
            Mutated copy with the shape of ``noise``.
        """
        device = noise.device
        num_mutations = int(torch.numel(noise) * self.mutation_rate)
        positions = torch.randperm(torch.numel(noise), device=device)[:num_mutations]
        perturbation = torch.randn(num_mutations, device=device) * self.mutation_strength

        flat = noise.clone().flatten()
        flat[positions] += perturbation
        return flat.view(noise.shape)


class UniformMutator(MutationFunction):
    """Replaces a random subset of the latent values by fresh Gaussian noise.

    Every value is replaced independently with probability ``mutation_rate`` by a draw from
    ``N(0, std^2)``.
    """

    def __init__(self, mutation_rate: float, std: float = 1.0) -> None:
        """
        Args:
            mutation_rate: Probability that a value is replaced.
            std: Standard deviation of the new values.
        """
        self.mutation_rate = mutation_rate
        self.std = std

    def mutate(self, noise: torch.Tensor) -> torch.Tensor:
        """Replace a random subset of the values of ``noise``.

        Args:
            noise: Noise tensor to mutate; it is not modified in place.

        Returns:
            Mutated copy with the shape of ``noise``.
        """
        fresh = torch.randn_like(noise) * self.std
        mask = torch.rand_like(noise) < self.mutation_rate
        return torch.where(mask, fresh, noise)
