"""Common interface of all image evaluators."""
from abc import ABC, abstractmethod
from typing import Any

from PIL import Image

EvaluationResult = dict[str, Any]
"""Score of one image: ``{"name": str, "score": float, "embeddings": {embedding_name: tensor}}``.

``embeddings`` is optional; embeddings returned here are written to the result archive.
"""


class BaseImageEvaluator(ABC):
    """Scores every image of a population.

    Attributes:
        name: Metric name; it becomes the CSV column ``fitness_<name>`` or ``model_<name>``.
    """

    name: str

    @abstractmethod
    def evaluate(self, images: list[Image.Image], prompt: str, update_reference: bool = False) -> list[EvaluationResult]:
        """Score a list of images.

        Args:
            images: Images of the current population, in population order.
            prompt: Prompt the images were generated from.
            update_reference: If ``True``, evaluators with a reference (e.g. a mean embedding) recompute it
                from ``images`` before scoring.

        Returns:
            One result per image, in the order of ``images``.
        """
