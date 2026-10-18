"""Text-to-image models of the model comparison."""
from evolutionary_creativity.models.text_to_image.base import GenerativeModel
from evolutionary_creativity.models.text_to_image.sdxl import StableDiffusionXLModel, StableDiffusionXLTurboModel
from evolutionary_creativity.models.text_to_image.transformer import (Flux2Dev, Flux2Klein4B, Flux2Klein9BKV,
                                                                      Krea2Turbo, ZImage)
from evolutionary_creativity.models.text_to_image.unet import (JuggernautXLModel, PlaygroundV25Model, RealVisXLModel,
                                                               SDTurboModel, SegmindVegaModel, SSD1BModel,
                                                               StableDiffusion15Model)
from experiments.model_comparison.config import MODEL_NAMES

TEXT_TO_IMAGE_MODELS: dict[str, tuple[type[GenerativeModel], dict]] = {
    "sdxl": (StableDiffusionXLModel, {"num_inference_steps": 30}),
    "sdxl_turbo": (StableDiffusionXLTurboModel, {}),
    "sd15": (StableDiffusion15Model, {}),
    "sd_turbo": (SDTurboModel, {}),
    "playground_v25": (PlaygroundV25Model, {}),
    "ssd_1b": (SSD1BModel, {}),
    "segmind_vega": (SegmindVegaModel, {}),
    "realvisxl_v5": (RealVisXLModel, {}),
    "juggernaut_xl_v9": (JuggernautXLModel, {}),
    "z_image": (ZImage, {}),
    "flux2_klein_4b": (Flux2Klein4B, {}),
    "flux2_klein_9b_kv": (Flux2Klein9BKV, {}),
    "krea_2_turbo": (Krea2Turbo, {}),
    "flux2_dev": (Flux2Dev, {}),
}
"""Model name -> (model class, keyword arguments). SDXL uses 30 steps as in the PartiPrompts experiment."""

assert list(TEXT_TO_IMAGE_MODELS) == MODEL_NAMES, "config.MODEL_NAMES and TEXT_TO_IMAGE_MODELS differ"


def load_model(name: str, device: str, cache_dir: str) -> GenerativeModel:
    """Load a text-to-image model of the comparison.

    Args:
        name: Model name as in the job database.
        device: Device of the pipeline.
        cache_dir: Hugging Face cache directory.

    Returns:
        The loaded model.

    Raises:
        ValueError: If ``name`` is unknown.
    """
    if name not in TEXT_TO_IMAGE_MODELS:
        raise ValueError(f"Unknown text-to-image model {name!r}. Available: {', '.join(TEXT_TO_IMAGE_MODELS)}")
    cls, kwargs = TEXT_TO_IMAGE_MODELS[name]
    return cls(device, cache_dir, **kwargs)
