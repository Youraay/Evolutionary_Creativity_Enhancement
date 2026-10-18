"""Novelty as mean distance to the nearest neighbours in the population."""
import torch
from PIL import Image
from torch.nn import functional as F

from evolutionary_creativity.evaluators.base import BaseImageEvaluator, EvaluationResult
from evolutionary_creativity.models.embeddings.base import EmbeddingModel


class KNearestNeighborEvaluator(BaseImageEvaluator):
    """Mean cosine distance ``1 - cos`` of every image to its ``n_neighbors`` nearest neighbours.

    The neighbours are searched in the reference set, which is replaced by the current embeddings whenever
    ``evaluate`` is called with ``update_reference=True`` (as fitness: every generation). The closest match
    of every image (the image itself) is skipped. The result equals that of
    ``sklearn.neighbors.NearestNeighbors(metric="cosine")``, which computed the analysed runs of
    ``experiments.fitness_function``.
    """

    def __init__(self, embedding_model: EmbeddingModel, device: str = "cuda", n_neighbors: int = 10) -> None:
        """
        Args:
            embedding_model: Image encoder that defines the embedding space.
            device: Device on which the distances are computed.
            n_neighbors: Number of neighbours ``k``.
        """
        self.embedding_model = embedding_model
        self.device = device
        self.n_neighbors = n_neighbors
        self.name = "LocalkNN"
        self.embedding_key = embedding_model.embedding_name
        self.reference: torch.Tensor | None = None

    def evaluate(self, images: list[Image.Image], prompt: str, update_reference: bool = False) -> list[EvaluationResult]:
        """Score every image by its mean distance to the ``k`` nearest reference embeddings (itself excluded).

        Args:
            images: Images of the current population.
            prompt: Unused; part of the evaluator interface.
            update_reference: Replace the reference set by the current embeddings before scoring.

        Returns:
            One result per image with the kNN score and its embedding.
        """
        current = self.embedding_model.image_features(images).to(self.device)
        if update_reference or self.reference is None:
            self.reference = F.normalize(current.float(), dim=1)
        distances = 1 - F.normalize(current.float(), dim=1) @ self.reference.T
        nearest = torch.topk(distances, self.n_neighbors + 1, dim=1, largest=False).values
        scores = nearest[:, 1:].mean(dim=1).tolist()
        return [{"name": self.name, "score": score,
                 "embeddings": {self.embedding_key: current[i].detach().unsqueeze(0)}}
                for i, score in enumerate(scores)]
