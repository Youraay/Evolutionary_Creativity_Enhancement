"""Novelty as distance to the mean embedding of a reference population."""
import logging
from collections import deque

import torch
from PIL import Image
from torch.nn import functional as F

from evolutionary_creativity.evaluators.base import BaseImageEvaluator, EvaluationResult
from evolutionary_creativity.models.embeddings.base import EmbeddingModel

logger = logging.getLogger(__name__)


class MaxMeanDivergenceEvaluator(BaseImageEvaluator):
    """Distance of every image to the mean embedding of a reference set.

    The score is ``(1 - cos(e, mean)) / 2`` in ``[0, 1]``. The reference mean is recomputed whenever
    ``evaluate`` is called with ``update_reference=True``: as fitness this happens every generation
    (distance to the current population), as analysis metric only in generation 0 (distance to the
    initial population). Optionally, embeddings of earlier generations are kept in an archive and enter
    the mean as well; with a threshold, only embeddings whose score exceeds it are archived.
    """

    def __init__(self, embedding_model: EmbeddingModel, device: str, archive_size: int = 0,
                 archive_threshold: float | None = None) -> None:
        """
        Args:
            embedding_model: Image encoder that defines the embedding space.
            device: Device on which the scores are computed.
            archive_size: Number of past embeddings kept in the archive; 0 disables the archive.
            archive_threshold: Only embeddings with a score above this value are archived; ``None``
                archives every embedding.
        """
        self.embedding_model = embedding_model
        self.device = device
        self.name = "LocalMaxMeanDivergence"
        self.embedding_key = embedding_model.embedding_name
        self.archive_size = archive_size
        self.archive_threshold = archive_threshold
        self.archive: deque[torch.Tensor] = deque(maxlen=archive_size)
        self.mean_embedding: torch.Tensor | None = None

    def evaluate(self, images: list[Image.Image], prompt: str, update_reference: bool = False) -> list[EvaluationResult]:
        """Score every image by its distance to the reference mean.

        Args:
            images: Images of the current population.
            prompt: Unused; part of the evaluator interface.
            update_reference: Recompute the reference mean from ``images`` (plus archive) before scoring.

        Returns:
            One result per image with the divergence score and its embedding.
        """
        current = self.embedding_model.image_features(images).to(self.device)

        if update_reference or self.mean_embedding is None:
            if self.archive_size > 0 and len(self.archive) > 0:
                reference = torch.cat([current, torch.cat(list(self.archive), dim=0).to(self.device)], dim=0)
                logger.debug("Reference mean from population (%d) and archive (%d)", len(current), len(self.archive))
            else:
                reference = current
                logger.debug("Reference mean from population (%d)", len(current))
            self.mean_embedding = torch.mean(reference, dim=0, keepdim=True).detach()

        similarity = F.cosine_similarity(current, self.mean_embedding)
        scores = (1 - (similarity + 1) / 2).tolist()

        results = []
        for i, score in enumerate(scores):
            embedding = current[i].detach().unsqueeze(0)
            results.append({"name": self.name, "score": score, "embeddings": {self.embedding_key: embedding}})
            if self.archive_size > 0 and (self.archive_threshold is None or score > self.archive_threshold):
                self.archive.append(embedding)
        return results
