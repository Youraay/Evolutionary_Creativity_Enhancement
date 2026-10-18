"""CLIP image and text encoder."""
import torch
from PIL import Image
from transformers import CLIPModel, CLIPProcessor

from evolutionary_creativity.models.embeddings.base import EmbeddingModel, output_tensor


class ClipEmbeddingModel(EmbeddingModel):
    """Projected CLIP image and text embeddings (ViT-B/32 by default), 512 dimensions."""

    embedding_name = "clip"

    def __init__(self, device: str, cache_dir: str, model: str = "openai/clip-vit-base-patch32") -> None:
        """
        Args:
            device: Device of the model.
            cache_dir: Hugging Face cache directory.
            model: Hugging Face model id.
        """
        self.device = device
        self.model = CLIPModel.from_pretrained(model, cache_dir=cache_dir)
        self.model.to(self.device)
        self.model.eval()
        self.processor = CLIPProcessor.from_pretrained(model, cache_dir=cache_dir)

    def _extract_image_features(self, images: list[Image.Image]) -> torch.Tensor:
        """Embed images.

        Args:
            images: Images to embed.

        Returns:
            Image embeddings of shape ``[len(images), 512]``.
        """
        inputs = self.processor(images=images, return_tensors="pt").to(self.device)
        with torch.inference_mode():
            return output_tensor(self.model.get_image_features(**inputs), "image_embeds")

    def extract_text_features(self, text: str) -> torch.Tensor:
        """Embed a text (truncated to 77 tokens).

        Args:
            text: Text to embed.

        Returns:
            Text embedding of shape ``[1, 512]``.
        """
        inputs = self.processor(text=[text], return_tensors="pt", padding=True, truncation=True).to(self.device)
        with torch.inference_mode():
            return output_tensor(self.model.get_text_features(**inputs), "text_embeds")
