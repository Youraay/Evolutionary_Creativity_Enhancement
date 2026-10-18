"""Data release: run archives and tables of all analysed runs, split into repositories of at most 50 GB.

Every release repository (``<DATA_RELEASE_DIR>/<repository>``) holds per experiment the candidate tables and
run configurations of ``data/results/<experiment>`` and the run archives, both with the relative layout of the
result directory (``<experiment>/<run path>/<run>.zip``). ``<EXPERIMENT>_RESULT_DIR`` can therefore point to
``<DATA_RELEASE_DIR>/<repository>/<experiment>``. The archives are built in two steps; each written archive is
checked before it is moved into place, and the result directories are only read.

1. Compaction without loss into ``<DATA_RELEASE_DIR>/_full`` (not part of any repository). Every tensor member
   (``.pt``) was saved as a view of the tensor of its whole batch or population and therefore contained the
   storage of that tensor; it is saved again with its own storage, with unchanged values, dtype and shape. All
   other members (images) are copied unchanged, and member names, order, timestamps and compression stay those
   of the original archive. Check: member names and order, CRC of every image, SHA-256 of the values of every
   tensor.
2. Selection of the members that are released (:data:`SELECTIONS`): every member of the generations 0, 10, 20
   and 30 (runs of more than 30 generations: also every 50th generation) and the members that the figures of
   the manuscript read from further generations. ``example_prompts`` is released completely (the PCA of its
   noise uses every generation). Check: member names and order, CRC of every member.

    python -m experiments.common.data_release plan
    python -m experiments.common.data_release build archive_size --workers 16    # finished archives are skipped
    python -m experiments.common.data_release manifest Evolutionary_Creativity_Enhancement_1
"""
import argparse
import csv
import hashlib
import io
import os
import re
import shutil
import time
import zipfile
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass, field
from pathlib import Path

from experiments.common.settings import REPO_ROOT, env_str, result_dir, table_dir

REPOSITORIES = {
    "Evolutionary_Creativity_Enhancement_1": ["example_prompts", "seed_transfer", "archive_size", "crossover_function",
                                              "fitness_function", "mutation_only", "operator_ablation"],
    "Evolutionary_Creativity_Enhancement_2": ["parti_prompts", "model_comparison", "fitness_embedding",
                                              "tournament_size", "long_runs", "think_then_generate"],
}
"""Release repositories and the experiments they contain: the runs with population 100 and those with population
60 (each repository below the 50 GB limit of G-Node GIN, no figure needs both)."""

RELEASE_DIR = Path(env_str("DATA_RELEASE_DIR", str(REPO_ROOT.parent / "Evolutionary_Creativity_Enhancement_Data")))
"""Root of the release repositories."""

FULL_DIR = RELEASE_DIR / "_full"
"""Archives compacted without loss, layout of the repositories (not part of any repository)."""

BUILD_DIR = RELEASE_DIR / "_build"
"""Records of the written archives: ``full/<experiment>.tsv`` (step 1), ``<experiment>.tsv`` (step 2)."""

GENERATIONS = (0, 10, 20, 30)
"""Generations whose members are released in every experiment."""

GENERATION = re.compile(r"(?:^|_)g(\d+)_")
MANIFEST = "MANIFEST.sha256"
RECORD_COLUMNS = ["path", "bytes", "sha256", "source", "source_bytes", "members", "tensors"]
CHUNK = 1 << 23


@dataclass(frozen=True)
class Selection:
    """Released members of the archives of an experiment.

    Attributes:
        complete: Release every member.
        step: Also release every member of every ``step``-th generation.
        folders: Folders (``images``, ``blip2``, ...) released in further generations, ``None`` for all.
    """

    complete: bool = False
    step: int | None = None
    folders: dict[str, tuple[int, ...] | None] = field(default_factory=dict)

    def keeps(self, name: str) -> bool:
        """Whether a member is released.

        Args:
            name: Member name, e.g. ``/blip2/blip2_g10_idn_0003_f0.1.pt``.

        Returns:
            True for members of the released generations and folders and for members without generation.
        """
        match = GENERATION.search(name.rsplit("/", 1)[-1])
        if self.complete or match is None:
            return True
        generation = int(match.group(1))
        if generation in GENERATIONS or (self.step and generation % self.step == 0):
            return True
        folder = name.strip("/").split("/")[0]
        return folder in self.folders and (self.folders[folder] is None or generation in self.folders[folder])


SELECTIONS = {
    "example_prompts": Selection(complete=True),
    "mutation_only": Selection(folders={"images": (8,)}),
    "parti_prompts": Selection(folders={"clip": None}),
    "long_runs": Selection(step=50, folders={"clip": None}),
    "think_then_generate": Selection(step=50, folders={"blip2": None}),
}
"""Exceptions from :class:`Selection` ``()`` (generations 0, 10, 20, 30) and the figures that need them:
``example_prompts`` noise PCA (Supp. Fig. 12B); ``mutation_only`` images of generation 8 (Supp. Fig. 9A);
``parti_prompts`` CLIP embeddings of the random baseline (Supp. Fig. 5B); ``long_runs`` images of generations 50
to 300 (Supp. Fig. 1D) and CLIP embeddings of the random baseline (Supp. Figs. 1D, 10); ``think_then_generate``
BLIP-2 embeddings of the recomputed divergence (Supp. Fig. 10)."""


def repository(experiment: str) -> str:
    """Release repository of an experiment.

    Args:
        experiment: Package name below ``experiments``.

    Returns:
        Name of the repository.

    Raises:
        KeyError: If the experiment belongs to no repository.
    """
    for name, experiments in REPOSITORIES.items():
        if experiment in experiments:
            return name
    raise KeyError(f"{experiment} belongs to no release repository")


def release_dir(experiment: str) -> Path:
    """Directory of an experiment in its release repository.

    Args:
        experiment: Package name below ``experiments``.

    Returns:
        ``<DATA_RELEASE_DIR>/<repository>/<experiment>``.
    """
    return RELEASE_DIR / repository(experiment) / experiment


def full_dir(experiment: str) -> Path:
    """Directory of the archives of an experiment compacted without loss.

    Args:
        experiment: Package name below ``experiments``.

    Returns:
        ``<DATA_RELEASE_DIR>/_full/<repository>/<experiment>``.
    """
    return FULL_DIR / repository(experiment) / experiment


def selection(experiment: str) -> Selection:
    """Released members of the archives of an experiment.

    Args:
        experiment: Package name below ``experiments``.

    Returns:
        The selection of :data:`SELECTIONS`, by default generations 0, 10, 20 and 30.
    """
    return SELECTIONS.get(experiment, Selection())


def run_paths(experiment: str) -> list[Path]:
    """Runs of an experiment: directories ``<run>`` with a candidate table ``<run>/<run>.csv`` in the repository.

    Args:
        experiment: Package name below ``experiments``.

    Returns:
        Run paths relative to ``data/results/<experiment>``.
    """
    tables = table_dir(experiment)
    return sorted(c.parent.relative_to(tables) for c in tables.rglob("*.csv") if c.stem == c.parent.name)


def archives(experiment: str) -> tuple[list[Path], list[Path]]:
    """Archives of the runs of an experiment.

    Args:
        experiment: Package name below ``experiments``.

    Returns:
        Archive paths relative to the experiment directory (``<run path>/<run>.zip``) of the runs whose
        original archive exists, and the run paths without one.
    """
    found, missing = [], []
    for run in run_paths(experiment):
        path = run / f"{run.name}.zip"
        if (result_dir(experiment) / path).exists():
            found.append(path)
        else:
            missing.append(run)
    return found, missing


def tensor_digest(tensor) -> str:
    """SHA-256 of dtype, shape and values of a tensor (independent of its storage).

    Args:
        tensor: Tensor.

    Returns:
        Hex digest.
    """
    import torch

    hasher = hashlib.sha256(f"{tensor.dtype}{tuple(tensor.shape)}".encode())
    hasher.update(tensor.detach().cpu().contiguous().reshape(-1).view(torch.uint8).numpy())
    return hasher.hexdigest()


def load_tensor(data: bytes):
    """Tensor of a ``.pt`` member.

    Args:
        data: Member content.

    Returns:
        The tensor.

    Raises:
        TypeError: If the member holds no single tensor.
    """
    import torch

    tensor = torch.load(io.BytesIO(data), map_location="cpu", weights_only=True)
    if not torch.is_tensor(tensor):
        raise TypeError(f"expected a tensor, got {type(tensor).__name__}")
    return tensor


def compact_tensor(data: bytes) -> tuple[bytes, str]:
    """Save the tensor of a ``.pt`` member again with its own storage.

    Args:
        data: Member content.

    Returns:
        New member content and the digest of the tensor.
    """
    import torch

    tensor = load_tensor(data)
    buffer = io.BytesIO()
    torch.save(tensor.clone(), buffer)
    return buffer.getvalue(), tensor_digest(tensor)


def copy_member(info: zipfile.ZipInfo) -> zipfile.ZipInfo:
    """Header of a member for a new archive with the name, timestamp, compression and attributes of ``info``.

    Args:
        info: Member of the source archive.

    Returns:
        New member header.
    """
    member = zipfile.ZipInfo(info.filename, info.date_time)
    member.compress_type = info.compress_type
    member.external_attr = info.external_attr
    member.create_system = info.create_system
    return member


def verify(path: Path, expected: list[tuple[str, int | str]]) -> None:
    """Check a written archive against its source.

    Args:
        path: Written archive.
        expected: Member name with the CRC of the source member, or for compacted tensors the tensor digest.

    Raises:
        ValueError: If names, order, contents or tensor storage differ.
    """
    with zipfile.ZipFile(path) as archive:
        infos = archive.infolist()
        if [i.filename for i in infos] != [name for name, _ in expected]:
            raise ValueError(f"{path}: member names or order differ")
        for info, (name, check) in zip(infos, expected):
            data = archive.read(info)
            if isinstance(check, int):
                if info.CRC != check:
                    raise ValueError(f"{path}: {name} differs")
                continue
            tensor = load_tensor(data)
            if tensor_digest(tensor) != check:
                raise ValueError(f"{path}: {name} has other values")
            if tensor.untyped_storage().nbytes() != tensor.numel() * tensor.element_size():
                raise ValueError(f"{path}: {name} still stores more than its values")


def sha256(path: Path) -> str:
    """SHA-256 of a file.

    Args:
        path: File.

    Returns:
        Hex digest.
    """
    hasher = hashlib.sha256()
    with open(path, "rb") as handle:
        while chunk := handle.read(CHUNK):
            hasher.update(chunk)
    return hasher.hexdigest()


def record(target: Path, source: Path, members: list[str]) -> dict:
    """Record of a written archive (see ``RECORD_COLUMNS``; ``path`` is absolute).

    Args:
        target: Written archive.
        source: Its source archive.
        members: Its member names.

    Returns:
        The record.
    """
    return {"path": str(target), "bytes": target.stat().st_size, "sha256": sha256(target), "source": str(source),
            "source_bytes": source.stat().st_size, "members": len(members),
            "tensors": sum(name.endswith(".pt") for name in members)}


def compact_archive(source: Path, target: Path) -> dict:
    """Step 1: write the archive with every tensor saved with its own storage, verify it, move it into place.

    Args:
        source: Original archive (only read).
        target: Compacted archive.

    Returns:
        Record of the archive.
    """
    import torch

    torch.set_num_threads(1)
    partial = target.with_name(f"{target.name}.partial")
    partial.parent.mkdir(parents=True, exist_ok=True)
    expected = []
    with zipfile.ZipFile(source) as original, zipfile.ZipFile(partial, "w", allowZip64=True) as compacted:
        for info in original.infolist():
            data = original.read(info)
            check = info.CRC
            if info.filename.endswith(".pt"):
                data, check = compact_tensor(data)
            compacted.writestr(copy_member(info), data)
            expected.append((info.filename, check))
    verify(partial, expected)
    partial.replace(target)
    return record(target, source, [name for name, _ in expected])


def select_archive(source: Path, target: Path, chosen: Selection) -> dict:
    """Step 2: write the archive with the released members, verify it, move it into place.

    An archive whose members are all released is linked (or copied) unchanged.

    Args:
        source: Archive compacted without loss.
        target: Released archive.
        chosen: Released members.

    Returns:
        Record of the archive.
    """
    partial = target.with_name(f"{target.name}.partial")
    partial.parent.mkdir(parents=True, exist_ok=True)
    partial.unlink(missing_ok=True)
    with zipfile.ZipFile(source) as full:
        infos = full.infolist()
        kept = [info for info in infos if chosen.keeps(info.filename)]
        if len(kept) == len(infos):
            try:
                os.link(source, partial)
            except OSError:
                shutil.copy2(source, partial)
        else:
            with zipfile.ZipFile(partial, "w", allowZip64=True) as released:
                for info in kept:
                    released.writestr(copy_member(info), full.read(info))
    verify(partial, [(info.filename, info.CRC) for info in kept])
    partial.replace(target)
    return record(target, source, [info.filename for info in kept])


def records_file(experiment: str, step: str) -> Path:
    """File of the records of one step.

    Args:
        experiment: Package name below ``experiments``.
        step: ``full`` (compaction) or ``release`` (selection).

    Returns:
        ``_build/full/<experiment>.tsv`` or ``_build/<experiment>.tsv``.
    """
    return BUILD_DIR / ("full" if step == "full" else "") / f"{experiment}.tsv"


def read_records(experiment: str, step: str) -> dict[str, dict]:
    """Records of the archives of an experiment already written in one step.

    Args:
        experiment: Package name below ``experiments``.
        step: ``full`` or ``release``.

    Returns:
        Record per path relative to the repository (``<experiment>/<run path>/<run>.zip``).
    """
    path = records_file(experiment, step)
    if not path.exists():
        return {}
    with open(path, newline="") as handle:
        return {row["path"]: row for row in csv.DictReader(handle, delimiter="\t")}


def append_record(experiment: str, step: str, entry: dict) -> None:
    """Append the record of a written archive.

    Args:
        experiment: Package name below ``experiments``.
        step: ``full`` or ``release``.
        entry: Record with ``path`` relative to the repository.
    """
    path = records_file(experiment, step)
    path.parent.mkdir(parents=True, exist_ok=True)
    new = not path.exists()
    with open(path, "a", newline="") as handle:
        writer = csv.DictWriter(handle, RECORD_COLUMNS, delimiter="\t")
        if new:
            writer.writeheader()
        writer.writerow(entry)


def copy_tables(experiment: str) -> int:
    """Copy the tables of ``data/results/<experiment>`` into the release (unchanged files are skipped).

    Args:
        experiment: Package name below ``experiments``.

    Returns:
        Number of copied files.
    """
    tables, target_root = table_dir(experiment), release_dir(experiment)
    copied = 0
    for source in sorted(p for p in tables.rglob("*") if p.is_file()):
        target = target_root / source.relative_to(tables)
        if target.exists() and target.stat().st_size == source.stat().st_size \
                and target.stat().st_mtime == source.stat().st_mtime:
            continue
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)
        copied += 1
    return copied


def run_step(experiment: str, step: str, workers: int) -> None:
    """Write every archive of one step that has no record yet.

    Args:
        experiment: Package name below ``experiments``.
        step: ``full`` (original -> ``_full``) or ``release`` (``_full`` -> repository).
        workers: Parallel processes.

    Raises:
        RuntimeError: If an archive could not be written.
    """
    found, _ = archives(experiment)
    source_root, target_root = (result_dir(experiment), full_dir(experiment)) if step == "full" \
        else (full_dir(experiment), release_dir(experiment))
    records = read_records(experiment, step)
    todo = []
    for path in found:
        key, target = str(Path(experiment) / path), target_root / path
        if not (target.exists() and key in records and int(records[key]["bytes"]) == target.stat().st_size):
            todo.append((source_root / path, target, key))
    todo.sort(key=lambda job: job[0].stat().st_size, reverse=True)
    print(f"[INFO] {experiment} ({step}): {len(found)} archives, {len(found) - len(todo)} done, {len(todo)} to "
          f"write ({sum(s.stat().st_size for s, _, _ in todo) / 1e9:.1f} GB) with {workers} workers", flush=True)
    start, failed, read, written = time.time(), [], 0, 0
    with ProcessPoolExecutor(max_workers=workers) as pool:
        futures = {(pool.submit(compact_archive, s, t) if step == "full"
                    else pool.submit(select_archive, s, t, selection(experiment))): (s, key) for s, t, key in todo}
        for number, future in enumerate(as_completed(futures), 1):
            source, key = futures[future]
            try:
                entry = future.result()
            except Exception as error:
                failed.append(source)
                print(f"[ERROR] {source}: {error!r}", flush=True)
                continue
            entry["path"] = key
            append_record(experiment, step, entry)
            read, written = read + entry["source_bytes"], written + entry["bytes"]
            hours = (time.time() - start) / 3600
            print(f"[{number}/{len(todo)}] {key}: {entry['source_bytes'] / 1e9:.2f} -> {entry['bytes'] / 1e9:.2f} GB"
                  f" | total {read / 1e9:.0f} -> {written / 1e9:.0f} GB, {read / 1e9 / max(hours, 1e-9):.0f} GB/h",
                  flush=True)
    if failed:
        raise RuntimeError(f"{len(failed)} archives of {experiment} failed in step {step}")


def build(experiment: str, workers: int) -> None:
    """Tables, compacted archives (step 1) and released archives (step 2) of one experiment.

    Args:
        experiment: Package name below ``experiments``.
        workers: Parallel processes.
    """
    print(f"[INFO] {experiment}: {copy_tables(experiment)} table files copied to {release_dir(experiment)}", flush=True)
    for run in archives(experiment)[1]:
        print(f"[WARN] {experiment}: no archive for {run}", flush=True)
    run_step(experiment, "full", workers)
    run_step(experiment, "release", workers)


def manifest(name: str) -> None:
    """Write ``MANIFEST.sha256`` of a release repository (``sha256sum -c`` format).

    Archives use the digest of their record when the size matches; all other files are hashed.

    Args:
        name: Release repository.

    Raises:
        RuntimeError: If an archive is unfinished.
    """
    root = RELEASE_DIR / name
    known = {}
    for experiment in REPOSITORIES[name]:
        known.update(read_records(experiment, "release"))
    lines = []
    for path in sorted(p for p in root.rglob("*") if p.is_file() and p.name != MANIFEST):
        relative = str(path.relative_to(root))
        if path.name.endswith(".partial"):
            raise RuntimeError(f"unfinished archive {path}")
        entry = known.get(relative)
        digest = entry["sha256"] if entry and int(entry["bytes"]) == path.stat().st_size else sha256(path)
        lines.append(f"{digest}  {relative}\n")
    (root / MANIFEST).write_text("".join(lines))
    print(f"[INFO] {name}: {len(lines)} files in {root / MANIFEST}")


def plan(names: list[str]) -> None:
    """Runs, archives and size per step of the experiments of the given repositories.

    Args:
        names: Release repositories.
    """
    print(f"{'repository':38s} {'experiment':22s} {'runs':>5s} {'zips':>5s} {'missing':>7s} {'original':>9s} "
          f"{'full':>9s} {'release':>9s}")
    for name in names:
        for experiment in REPOSITORIES[name]:
            found, missing = archives(experiment)
            sizes = [sum((root / p).stat().st_size for p in found if (root / p).exists()) / 1e9
                     for root in (result_dir(experiment), full_dir(experiment), release_dir(experiment))]
            print(f"{name:38s} {experiment:22s} {len(found) + len(missing):5d} {len(found):5d} {len(missing):7d} "
                  + " ".join(f"{s:8.1f}G" for s in sizes))


def main() -> None:
    """Command line entry point."""
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("plan").add_argument("repositories", nargs="*", choices=list(REPOSITORIES))
    command = commands.add_parser("build")
    command.add_argument("experiments", nargs="+")
    command.add_argument("--workers", type=int, default=8)
    commands.add_parser("manifest").add_argument("repositories", nargs="+", choices=list(REPOSITORIES))
    args = parser.parse_args()
    if args.command == "plan":
        plan(args.repositories or list(REPOSITORIES))
    elif args.command == "build":
        for experiment in args.experiments:
            build(experiment, args.workers)
    else:
        for name in args.repositories:
            manifest(name)


if __name__ == "__main__":
    main()
