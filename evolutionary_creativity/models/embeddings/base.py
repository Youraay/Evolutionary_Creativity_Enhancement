"""Common interface of the image embedding models."""
from abc import ABC, abstractmethod

import torch
from PIL import Image


def output_tensor(output, key: str) -> torch.Tensor:
    """Return the embedding tensor from a model output that may be a tensor or a ``ModelOutput``.

    Args:
        output: Return value of ``get_image_features`` / ``get_text_features``.
        key: Attribute holding the embedding if ``output`` is not a tensor.

    Returns:
        The embedding tensor.
    """
    if isinstance(output, torch.Tensor):
        return output
    return getattr(output, key, getattr(output, "pooler_output", output[0]))


class EmbeddingModel(ABC):
    """Image encoder with a one-entry cache.

    The pipeline passes the same list of images to every evaluator of a generation. When several
    evaluators share one embedding model, the encoder therefore runs only once per generation: the
    cache keeps the last list (by identity) and its features. The next generation builds a new list,
    which invalidates the cache.

    Attributes:
        embedding_name: Short name of the embedding space; used as folder name in the result archive.
    """

    embedding_name: str
    _cached_images: list[Image.Image] | None = None
    _cached_features: torch.Tensor | None = None

    def image_features(self, images: list[Image.Image]) -> torch.Tensor:
        """Embed a list of images, reusing the result for the same list object.

        Args:
            images: Images to embed.

        Returns:
            Embedding matrix of shape ``[len(images), D]``.
        """
        if images is self._cached_images:
            return self._cached_features
        features = self._extract_image_features(images)
        self._cached_images = images
        self._cached_features = features
        return features

    def extract_text_features(self, text: str) -> torch.Tensor:
        """Embed a text into the image embedding space (only for image-text models).

        Args:
            text: Text to embed.

        Returns:
            Text embedding of shape ``[1, D]``.

        Raises:
            NotImplementedError: If the model has no text encoder.
        """
        raise NotImplementedError(f"{type(self).__name__} has no text encoder")

    @abstractmethod
    def _extract_image_features(self, images: list[Image.Image]) -> torch.Tensor:
        """Run the image encoder.

        Args:
            images: Images to embed.

        Returns:
            Embedding matrix of shape ``[len(images), D]``.
        """
