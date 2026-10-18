"""Crossover operators."""
import math
from abc import ABC, abstractmethod
from dataclasses import dataclass

import torch


class CrossoverFunction(ABC):
    """Combines two parent noise tensors into one child."""

    @abstractmethod
    def crossover(self, parent_1: torch.Tensor, parent_2: torch.Tensor) -> torch.Tensor:
        """Create a child from two parents.

        Args:
            parent_1: First parent noise.
            parent_2: Second parent noise of the same shape.

        Returns:
            Child noise with the shape of the parents.
        """


@dataclass
class UniformCrossover(CrossoverFunction):
    """Element-wise uniform crossover: every latent value is taken from either parent.

    The mask is drawn with the global CPU generator of torch and then moved to ``device``, so runs are
    reproducible from ``torch.manual_seed``.

    Attributes:
        swap_rate: Probability that a value is taken from ``parent_1``.
        device: Device on which the child is assembled.
    """

    swap_rate: float = 0.5
    device: str = "cuda"
    name = "UniformCrossover"

    def crossover(self, parent_1: torch.Tensor, parent_2: torch.Tensor) -> torch.Tensor:
        """Create a child by picking every value from one of the two parents.

        Args:
            parent_1: First parent noise, e.g. ``[1, C, H, W]``.
            parent_2: Second parent noise of the same shape.

        Returns:
            Child noise on ``device`` with the shape of the parents.

        Raises:
            AssertionError: If the parents differ in shape.
        """
        assert parent_1.shape == parent_2.shape, "Noises must have the same size for crossover."
        parent_1 = parent_1.to(self.device)
        parent_2 = parent_2.to(self.device)
        mask = (torch.rand(parent_1.shape) < self.swap_rate).to(self.device)
        return torch.where(mask, parent_1, parent_2)


@dataclass
class UniformPixelCrossover(CrossoverFunction):
    """Uniform crossover per latent pixel: all channels of a position are taken from the same parent.

    The mask is drawn on the device of the parents (CUDA generator for CUDA tensors).

    Attributes:
        swap_rate: Probability that a position is taken from ``parent_1``.
    """

    swap_rate: float = 0.5
    name = "UniformPixelCrossover"

    def crossover(self, parent_1: torch.Tensor, parent_2: torch.Tensor) -> torch.Tensor:
        """Create a child by picking every latent position (all channels) from one of the two parents.

        Args:
            parent_1: First parent noise ``[..., C, H, W]``.
            parent_2: Second parent noise of the same shape.

        Returns:
            Child noise with the shape of the parents.

        Raises:
            AssertionError: If the parents differ in shape.
        """
        assert parent_1.shape == parent_2.shape, "Parents must have the same size for crossover."
        mask_shape = list(parent_1.shape)
        mask_shape[-3] = 1
        mask = torch.rand(mask_shape, device=parent_1.device) < self.swap_rate
        return torch.where(mask, parent_1, parent_2)


class NPointCrossover(CrossoverFunction):
    """n-point crossover with equally spaced cut points over the flattened latent positions.

    The ``H * W`` positions are split into ``n_points + 1`` segments of equal length (the last one takes
    the remainder); segments alternate between the parents, starting with ``parent_1``. All channels of a
    position come from the same parent. With equal spacing, the "pixel" and the channel-wise variant of the
    crossover comparison (``experiments.crossover_function``) produce identical children.
    """

    def __init__(self, n_points: int) -> None:
        """
        Args:
            n_points: Number of cut points.
        """
        self.n_points = n_points
        self.name = f"{n_points}_PointEqualCrossoverFunction"

    def crossover(self, parent_1: torch.Tensor, parent_2: torch.Tensor) -> torch.Tensor:
        """Create a child from alternating segments of the two parents.

        Args:
            parent_1: First parent noise ``[1, C, H, W]`` or ``[C, H, W]``.
            parent_2: Second parent noise of the same shape.

        Returns:
            Child noise with the shape of the parents.

        Raises:
            AssertionError: If the parents differ in shape.
        """
        assert parent_1.shape == parent_2.shape, "Parents must have the same size for crossover."
        batched = parent_1.ndim == 4
        if not batched:
            parent_1, parent_2 = parent_1.unsqueeze(0), parent_2.unsqueeze(0)
        batch, _, height, width = parent_1.shape
        length = height * width
        segment = length // (self.n_points + 1)
        cuts = (torch.arange(1, self.n_points + 1, device=parent_1.device) * segment).view(1, self.n_points, 1)
        positions = torch.arange(length, device=parent_1.device).view(1, 1, length)
        from_parent_1 = ((positions >= cuts).sum(dim=1) % 2) == 0
        child = torch.where(from_parent_1.view(1, 1, height, width).expand(batch, 1, height, width), parent_1, parent_2)
        return child if batched else child.squeeze(0)


@dataclass
class ArithmeticCrossover(CrossoverFunction):
    """Weighted mean of the parents, rescaled to keep unit variance for independent Gaussian parents.

    ``child = (alpha * p1 + (1 - alpha) * p2) / sqrt(alpha^2 + (1 - alpha)^2)``.

    Attributes:
        alpha: Weight of ``parent_1``.
    """

    alpha: float = 0.5
    name = "ArithmeticCrossoverFunction"

    def crossover(self, parent_1: torch.Tensor, parent_2: torch.Tensor) -> torch.Tensor:
        """Combine the parents arithmetically.

        Args:
            parent_1: First parent noise.
            parent_2: Second parent noise of the same shape (moved to the device of ``parent_1``).

        Returns:
            Child noise with the shape of the parents.
        """
        parent_2 = parent_2.to(parent_1.device)
        child = self.alpha * parent_1 + (1 - self.alpha) * parent_2
        return child / math.sqrt(self.alpha ** 2 + (1 - self.alpha) ** 2)
