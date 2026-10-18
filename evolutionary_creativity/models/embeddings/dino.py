"""DINOv2 image encoder."""
import torch
from PIL import Image
from transformers import AutoImageProcessor, AutoModel

from evolutionary_creativity.models.embeddings.base import EmbeddingModel


class DinoEmbeddingModel(EmbeddingModel):
    """DINOv2: self-supervised vision-only encoder; embedding = layer-normalised CLS token, 768 dimensions."""

    embedding_name = "dino"

    def __init__(self, device: str, cache_dir: str, model: str = "facebook/dinov2-base") -> None:
        """
        Args:
            device: Device of the model.
            cache_dir: Hugging Face cache directory.
            model: Hugging Face model id.
        """
        self.device = device
        self.model = AutoModel.from_pretrained(model, cache_dir=cache_dir, use_safetensors=True)
        self.model.to(self.device)
        self.model.eval()
        self.processor = AutoImageProcessor.from_pretrained(model, cache_dir=cache_dir)

    def _extract_image_features(self, images: list[Image.Image]) -> torch.Tensor:
        """Embed images.

        ``pooler_output`` is a view on the full token output (``N x 257 x 768``). It is cloned so that
        ``torch.save`` stores only the CLS embedding (about 0.2 MB instead of 47 MB per image).

        Args:
            images: Images to embed.

        Returns:
            CLS embeddings of shape ``[len(images), 768]``.
        """
        inputs = self.processor(images=images, return_tensors="pt").to(self.device)
        with torch.inference_mode():
            output = self.model(**inputs)
        return output.pooler_output.clone()
