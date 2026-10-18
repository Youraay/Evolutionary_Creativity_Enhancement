"""BLIP-2 vision encoder."""
import torch
from PIL import Image
from transformers import AutoProcessor, Blip2Model

from evolutionary_creativity.models.embeddings.base import EmbeddingModel


class Blip2EmbeddingModel(EmbeddingModel):
    """Pooled output of the BLIP-2 vision encoder (EVA ViT-g), 1408 dimensions, float16."""

    embedding_name = "blip2"

    def __init__(self, device: str, cache_dir: str, model: str = "Salesforce/blip2-opt-2.7b") -> None:
        """
        Args:
            device: Device of the model.
            cache_dir: Hugging Face cache directory.
            model: Hugging Face model id.
        """
        self.device = device
        self.model = Blip2Model.from_pretrained(model, cache_dir=cache_dir, use_safetensors=True,
                                                torch_dtype=torch.float16).to(self.device)
        self.model.eval()
        self.processor = AutoProcessor.from_pretrained(model, cache_dir=cache_dir)

    def _extract_image_features(self, images: list[Image.Image]) -> torch.Tensor:
        """Embed images with the vision encoder.

        Args:
            images: Images to embed.

        Returns:
            Pooled vision features of shape ``[len(images), 1408]``.
        """
        inputs = self.processor(images=images, return_tensors="pt").to(self.device)
        with torch.inference_mode():
            return self.model.get_image_features(**inputs).pooler_output
