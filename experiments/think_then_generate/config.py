"""Paths of the Think-Then-Generate experiment (no heavy imports, also used by the analysis)."""
from experiments.common.settings import DATA_DIR, result_dir

RESULT_DIR = result_dir("think_then_generate")
REWRITES = DATA_DIR / "prompts" / "t2g_rewritten_prompts.json"
"""Rewritten prompts: ``{"_source": {...}, "prompts": {original: {"revised_prompt", "reasoning", ...}}}``."""
