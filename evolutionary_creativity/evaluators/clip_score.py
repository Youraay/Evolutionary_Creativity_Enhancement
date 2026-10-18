"""Prompt fidelity as CLIP score."""
import torch
from PIL import Image

from evolutionary_creativity.evaluators.base import BaseImageEvaluator, EvaluationResult
from evolutionary_creativity.models.embeddings.base import EmbeddingModel


class CLIPScoreEvaluator(BaseImageEvaluator):
    """CLIP score ``scale * max(cos(image, text), 0)`` (Hessel et al., 2021).

    Works with every embedding model that also embeds text (CLIP, SigLIP). The prompt is prefixed with
    ``"A photo depicts "`` as in the original CLIPScore definition.
    """

    def __init__(self, embedding_model: EmbeddingModel, device: str = "cuda", prefix: str = "A photo depicts ",
                 use_prefix: bool = True, scale_factor: float = 2.5) -> None:
        """
        Args:
            embedding_model: Image-text model with ``extract_text_features``.
            device: Device of the model (kept for interface symmetry).
            prefix: Text put in front of the prompt.
            use_prefix: Whether to add ``prefix`` (it is never added twice).
            scale_factor: Factor ``w`` of the CLIP score.
        """
        self.embedding_model = embedding_model
        self.device = device
        self.name = "CLIPScore"
        self.embedding_key = embedding_model.embedding_name
        self.prefix = prefix
        self.use_prefix = use_prefix
        self.scale_factor = scale_factor
        self._cached_prompt: str | None = None
        self._text_features: torch.Tensor | None = None

    def evaluate(self, images: list[Image.Image], prompt: str, update_reference: bool = False) -> list[EvaluationResult]:
        """Score every image against ``prompt``; the text embedding is cached per prompt.

        Args:
            images: Images of the current population.
            prompt: Prompt the images are compared with.
            update_reference: Unused; part of the evaluator interface.

        Returns:
            One result per image with the CLIP score and the unnormalised image embedding.
        """
        raw_features = self.embedding_model.image_features(images)
        image_features = raw_features / raw_features.norm(p=2, dim=-1, keepdim=True)

        if self._cached_prompt != prompt or self._text_features is None:
            text = f"{self.prefix}{prompt}" if self.use_prefix and not prompt.startswith(self.prefix) else prompt
            text_features = self.embedding_model.extract_text_features(text)
            self._text_features = text_features / text_features.norm(p=2, dim=-1, keepdim=True)
            self._cached_prompt = prompt

        similarity = (self._text_features @ image_features.T).squeeze(0)
        clip_scores = self.scale_factor * torch.clamp(similarity, min=0.0)
        scores = [clip_scores.item()] if clip_scores.dim() == 0 else clip_scores.tolist()

        return [{"name": self.name, "score": score, "embeddings": {self.embedding_key: raw_features[i].detach().unsqueeze(0)}}
                for i, score in enumerate(scores)]
