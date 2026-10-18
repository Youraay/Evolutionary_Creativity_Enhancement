"""Shared command-line, integrity and output conventions for the Gemma analyses."""

import argparse
import json
import platform
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd
import scipy

from analysis.common.gemma.data import DATA, require, sha256
from analysis.common.paths import PLOTS_DIR, REPO_ROOT
from analysis.common.style import set_preview_dir


def arguments(description, name):
    parser = argparse.ArgumentParser(
        description=description, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--preview", type=Path, help="also write PNG previews into this directory")
    parser.add_argument("--output", type=Path, default=PLOTS_DIR / name)
    parser.add_argument("--data", type=Path, default=DATA, help="Gemma input directory")
    parser.add_argument(
        "--deep", action="store_true", help="rebuild all post-hoc summaries from individual ratings"
    )
    parser.add_argument(
        "--verify-only", action="store_true", help="check inputs and statistics without writing plots"
    )
    args = parser.parse_args()
    args.data = args.data.expanduser().resolve()
    args.output = args.output.expanduser().resolve()
    require(
        not args.output.is_relative_to(args.data) and not args.data.is_relative_to(args.output),
        "Output directory must not overlap the source data",
    )
    set_preview_dir(args.preview)
    return args


def check_hashes(root=DATA, groups=None):
    """Verify the relevant source files; no generated files are treated as inputs."""
    root = Path(root).resolve()
    manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    matched = 0
    for relative, expected in manifest["sha256"].items():
        if groups and Path(relative).parts[0] not in groups:
            continue
        path = (root / relative).resolve()
        require(path.is_relative_to(root), "Unsafe manifest path")
        require(path.is_file() and sha256(path) == expected, f"Hash mismatch: {relative}")
        matched += 1
    require(matched > 0, "Empty source manifest selection")
    return manifest


def finish(args, module, stem, before, groups):
    """Record inputs, code and generated file hashes alongside each figure."""
    require(before == check_hashes(args.data, groups), "Inputs changed during rendering")
    code = sorted((REPO_ROOT / "analysis/common/gemma").glob("*.py"))
    code += [
        REPO_ROOT / f"analysis/{module}.py",
        REPO_ROOT / "analysis/common/style.py",
        REPO_ROOT / "analysis/common/paths.py",
        REPO_ROOT / "experiments/common/settings.py",
    ]
    report = {
        "status": "complete",
        "module": f"analysis.{module}",
        "no_inference": True,
        "deep_raw_score_rebuild": args.deep and module == "gemma_ratings",
        "python": platform.python_version(),
        "numpy": np.__version__,
        "pandas": pd.__version__,
        "scipy": scipy.__version__,
        "matplotlib": matplotlib.__version__,
        "source_manifest_sha256": sha256(args.data / "manifest.json"),
        "code_sha256": {p.relative_to(REPO_ROOT).as_posix(): sha256(p) for p in code},
        "outputs_sha256": {
            p.name: sha256(p)
            for p in sorted(args.output.iterdir())
            if p.is_file() and p.name != "render_manifest.json"
        },
    }
    (args.output / "render_manifest.json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8"
    )
    print(f"[INFO] complete: {args.output / stem}", flush=True)
