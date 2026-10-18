"""Names of result directories (no heavy imports, also used by the analysis)."""
import re

PROMPT_SLUG_MAX = 100


def slugify_prompt(prompt: str, max_length: int = PROMPT_SLUG_MAX) -> str:
    """Turn a prompt into a safe path component (spaces and punctuation become ``_``).

    The slug is truncated because a path component may have at most 255 bytes: the longest PartiPrompt
    gives a slug of 230 characters, which together with seed, timestamp and extension would exceed the
    limit. At 100 characters all prompts of the selection are still unique.

    Args:
        prompt: Text prompt.
        max_length: Maximum slug length.

    Returns:
        The slug, or ``"prompt"`` if nothing is left.
    """
    slug = re.sub(r"[^\w\-]+", "_", prompt, flags=re.UNICODE).strip("_")
    slug = re.sub(r"_+", "_", slug)
    return slug[:max_length].rstrip("_") or "prompt"
