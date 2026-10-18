"""Image embedding models used by the evaluators."""
from evolutionary_creativity.models.embeddings.base import EmbeddingModel
from evolutionary_creativity.models.embeddings.blip2 import Blip2EmbeddingModel
from evolutionary_creativity.models.embeddings.clip import ClipEmbeddingModel
from evolutionary_creativity.models.embeddings.dino import DinoEmbeddingModel
from evolutionary_creativity.models.embeddings.siglip import SiglipEmbeddingModel

__all__ = ["EmbeddingModel", "Blip2EmbeddingModel", "ClipEmbeddingModel", "DinoEmbeddingModel",
           "SiglipEmbeddingModel"]
