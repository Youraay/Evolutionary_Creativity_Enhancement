"""Diversity of a population as mean pairwise embedding distance."""
import torch
from PIL import Image
from torch.nn import functional as F

from evolutionary_creativity.evaluators.base import BaseImageEvaluator, EvaluationResult
from evolutionary_creativity.models.embeddings.base import EmbeddingModel


class DiversityWithinPopulationEvaluator(BaseImageEvaluator):
    """Mean distance ``(1 - cos) / 2`` of every image to all other images of the same generation."""

    def __init__(self, embedding_model: EmbeddingModel, device: str = "cuda") -> None:
        """
        Args:
            embedding_model: Image encoder that defines the embedding space.
            device: Device on which the distances are computed.
        """
        self.embedding_model = embedding_model
        self.device = device
        self.name = "DiversityWithinPopulation"
        self.embedding_key = embedding_model.embedding_name

    def evaluate(self, images: list[Image.Image], prompt: str, update_reference: bool = False) -> list[EvaluationResult]:
        """Score every image by its mean distance to the other images.

        Args:
            images: Images of the current population.
            prompt: Unused; part of the evaluator interface.
            update_reference: Unused; the reference is always the current population.

        Returns:
            One result per image; a population of one image gets a single score of 0.
        """
        current = self.embedding_model.image_features(images).to(self.device)
        n = current.shape[0]
        if n <= 1:
            return [{"name": self.name, "score": 0.0,
                     "embeddings": {self.embedding_key: current[0].detach().unsqueeze(0)}}]

        normalized = F.normalize(current, p=2, dim=1)
        distances = 1 - ((normalized @ normalized.t() + 1) / 2)
        scores = (distances.sum(dim=1) / (n - 1)).tolist()

        return [{"name": self.name, "score": score,
                 "embeddings": {self.embedding_key: current[i].detach().unsqueeze(0)}}
                for i, score in enumerate(scores)]
