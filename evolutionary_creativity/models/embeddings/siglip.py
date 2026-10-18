"""SigLIP 2 image and text encoder."""
import torch
from PIL import Image
from transformers import AutoModel, AutoProcessor

from evolutionary_creativity.models.embeddings.base import EmbeddingModel, output_tensor


class SiglipEmbeddingModel(EmbeddingModel):
    """SigLIP 2: image-text model trained with a sigmoid instead of a softmax contrastive loss, 768 dimensions."""

    embedding_name = "siglip"

    def __init__(self, device: str, cache_dir: str, model: str = "google/siglip2-base-patch16-224") -> None:
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
        self.processor = AutoProcessor.from_pretrained(model, cache_dir=cache_dir)

    def _extract_image_features(self, images: list[Image.Image]) -> torch.Tensor:
        """Embed images.

        Args:
            images: Images to embed.

        Returns:
            Pooled image embeddings of shape ``[len(images), 768]``.
        """
        inputs = self.processor(images=images, return_tensors="pt").to(self.device)
        with torch.inference_mode():
            return output_tensor(self.model.get_image_features(**inputs), "image_embeds")

    def extract_text_features(self, text: str) -> torch.Tensor:
        """Embed a text, padded to 64 tokens as during SigLIP training.

        Args:
            text: Text to embed.

        Returns:
            Text embedding of shape ``[1, 768]``.
        """
        inputs = self.processor(text=[text], return_tensors="pt", padding="max_length", max_length=64,
                                truncation=True).to(self.device)
        with torch.inference_mode():
            return output_tensor(self.model.get_text_features(**inputs), "text_embeds")
