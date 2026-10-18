"""Experiments whose runs differ in the operators, rates or fitness function of the genetic algorithm.

These experiments use population 100, SDXL in float16 with 50 denoising steps and batch size 5 (:data:`STANDARD`).
Every job row stores the complete configuration of its run, so one worker serves all of them. Operators
and fitness are given as short specifications:

    crossover  ``uniform`` | ``uniform_pixel`` | ``n_point:<n>`` | ``arithmetic``
    mutator    ``gaussian:<fraction>:<strength>`` | ``uniform:<rate>``
    fitness    ``mmd`` | ``mmd_archive:<size>:<threshold>`` | ``knn:<k>``

Transfer jobs (``source_prompt`` set, ``num_generations`` 0) do not evolve: the noise of generations 0, 10,
20 and 30 of a run of ``source_prompt`` is rendered and evaluated with ``prompt``.

Run directories are named ``<variant>_Seed_<label>_<timestamp>`` (transfer: ``<source>_Seed_<label>_to_<target>``).
The analysed runs were started with one script per run; ``create_jobs --register`` lists them in the job
database with status ``DONE``. Their candidate tables use the column names ``global_score`` (distance to the
mean of generation 0), ``fitness``, ``diversity_score`` and ``prompt_fidelty`` (cosine similarity of the CLIP
embeddings of image and prompt); runs of the worker below record ``model_LocalMaxMeanDivergence``,
``model_DiversityWithinPopulation`` and ``model_CLIPScore`` instead (see ``analysis.common.runs.read_run``).
"""
import argparse
import io
import json
import logging
import re
import sqlite3
import zipfile
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

from experiments.common.job_queue import JobQueue

logger = logging.getLogger(__name__)

SEED_SETS = {
    "1-3": {"1": "8340687860103266623", "2": "6477699659723909009", "3": "12951769831226840325"},
    "4-6": {"4": "14115672566372762410", "5": "12876743012905148135", "6": "12055870704752283052"},
    "7-9": {"7": "3263689803816884559", "8": "8307662703890363777", "9": "17278950119299660979"},
}
"""Seed label -> seed. Every variant uses one set of three seeds."""

STANDARD = {"crossover": "uniform", "crossover_rate": 0.9, "mutator": "gaussian:0.1:0.2", "mutation_rate": 0.05,
            "elite_size": 1, "fitness": "mmd", "population_size": 100, "num_generations": 30, "batch_size": 5,
            "inference_steps": 50}
"""Standard configuration; every variant changes some entries."""

TRANSFER_GENERATIONS = [0, 10, 20, 30]
"""Generations of the source run that are rendered with the target prompt."""

COLUMNS = [
    ("variant", "TEXT NOT NULL"),
    ("run_name", "TEXT NOT NULL"),
    ("prompt", "TEXT NOT NULL"),
    ("seed_label", "TEXT NOT NULL"),
    ("seed", "TEXT NOT NULL"),
    ("source_prompt", "TEXT"),
    ("crossover", "TEXT NOT NULL"),
    ("crossover_rate", "REAL NOT NULL"),
    ("mutator", "TEXT NOT NULL"),
    ("mutation_rate", "REAL NOT NULL"),
    ("elite_size", "INTEGER NOT NULL"),
    ("fitness", "TEXT NOT NULL"),
    ("population_size", "INTEGER NOT NULL"),
    ("num_generations", "INTEGER NOT NULL"),
    ("batch_size", "INTEGER NOT NULL"),
    ("inference_steps", "INTEGER NOT NULL"),
    ("note", "TEXT"),
]
"""Columns of the job table. For transfer jobs ``num_generations`` is 0 and ``source_prompt`` names the run
whose noise is rendered with ``prompt``."""

RUN_PATTERN = re.compile(r"^(?P<name>.+?)_Seed_(?P<label>\d+)(?:_(?P<job>\d+)_(?P<line>\d+))?$")
"""Directory names of the registered runs: ``<variant>_Seed_<label>`` with optional Slurm job and line."""
TRANSFER_PATTERN = re.compile(r"^(?P<source>.+)_Seed_(?P<label>\d+)_to_(?P<target>.+)$")

OPERATOR_NAMES = {
    "crossover": {"uniform": {"UniformCrossover"}, "uniform_pixel": {"UniformPixelCrossover"},
                  "n_point": {"NPointEqualCrossoverFunction", "NPointEqualPixelCrossoverFunction"},
                  "arithmetic": {"ArithmeticRenormedCrossoverFunction"}},
    "mutator": {"gaussian": {"UniformGaussianMutator"}, "uniform": {"UniformMutator"}},
    "fitness": {"mmd": {"LocalMaxMeanDivergenceEvaluator", "LocalArchivMaxMeanDivergenceEvaluator"},
                "mmd_archive": {"LocalArchivMaxMeanDivergenceEvaluator"}, "knn": {"LocalkNNEvaluator"}},
}
"""Specification -> class names in the configuration files (``<run>.json``) of the registered runs."""
CONFIG_KEYS = {"crossover": "crossover_function", "mutator": "mutator", "fitness": "evaluator"}
NUMERIC_KEYS = {"crossover_rate": "crossover_rate", "mutation_rate": "initial_mutation_rate",
                "elite_size": "elitism_count", "population_size": "population_size",
                "num_generations": "num_generations", "batch_size": "batch_size"}


def prompt_name(prompt: str) -> str:
    """Run name prefix of a prompt, e.g. ``"a coffee mug"`` -> ``"a_coffee_mug"``.

    Args:
        prompt: Text prompt.

    Returns:
        The prompt with spaces replaced by underscores.
    """
    return prompt.replace(" ", "_")


@dataclass
class Variant:
    """One configuration of an experiment.

    Attributes:
        name: Name prefix of the run directories (``<name>_Seed_<label>``).
        prompt: Prompt of the runs.
        seeds: Key of :data:`SEED_SETS`.
        overrides: Deviations from :data:`STANDARD`.
        rerun: Whether ``create_jobs`` schedules the variant by default (``False`` for variants that duplicate
            another variant).
    """

    name: str
    prompt: str = "a cat"
    seeds: str = "1-3"
    overrides: dict = field(default_factory=dict)
    rerun: bool = True


@dataclass
class VariantExperiment:
    """Paths and variants of one experiment.

    Attributes:
        name: Package name below ``experiments``.
        variants: All variants of the experiment.
        db_file: Job database.
        result_dir: Directory of the run directories.
        transfer_source_dir: Directory of the runs whose noise transfer jobs render (only for transfer experiments).
        source_corrections: Prefix of a transfer run directory (``<source>_Seed_<label>``) -> prompt of the run
            whose noise it actually rendered, for directories named after the wrong source.
    """

    name: str
    variants: list[Variant]
    db_file: Path
    result_dir: Path
    transfer_source_dir: Path | None = None
    source_corrections: dict[str, str] = field(default_factory=dict)

    def job_queue(self, db_file: Path | None = None) -> JobQueue:
        """Return the job database.

        Args:
            db_file: Database path; by default :attr:`db_file`.
        """
        return JobQueue(str(db_file or self.db_file), COLUMNS)

    def find_variant(self, name: str) -> Variant | None:
        """Variant whose name equals ``name`` (case-insensitive), or ``None``."""
        return next((v for v in self.variants if v.name.lower() == name.lower()), None)


def job_row(variant: Variant, seed_label: str, seed: str | None = None, note: str = "") -> dict:
    """Build the job row of one run of a variant.

    Args:
        variant: Variant of the experiment.
        seed_label: Seed label, e.g. ``"1"``.
        seed: Seed; by default the seed of the label in the variant's seed set.
        note: Free text.

    Returns:
        Dict with a value for every column of :data:`COLUMNS`.
    """
    settings = {**STANDARD, **variant.overrides}
    source = settings.pop("source_prompt", None)
    run_name = (f"{prompt_name(source)}_Seed_{seed_label}_to_{prompt_name(variant.prompt)}" if source
                else f"{variant.name}_Seed_{seed_label}")
    return {"variant": variant.name, "run_name": run_name, "prompt": variant.prompt, "seed_label": seed_label,
            "seed": seed or SEED_SETS[variant.seeds][seed_label], "source_prompt": source, "note": note, **settings}


def transfer_variants(prompts: list[str]) -> list[Variant]:
    """Transfer variants for all ordered pairs of prompts (including a prompt with itself).

    Args:
        prompts: Source and target prompts.

    Returns:
        One variant ``<source>_to_<target>`` per pair.
    """
    return [Variant(f"{prompt_name(source)}_to_{prompt_name(target)}", prompt=target,
                    overrides={"num_generations": 0, "source_prompt": source})
            for source in prompts for target in prompts]


# --- registration of finished runs -------------------------------------------------------------------------


def check_config(row: dict, config: dict, run: str) -> list[str]:
    """Compare a job row with the configuration file of a run.

    Args:
        row: Job row built from the variant definition.
        config: Content of ``<run>.json``.
        run: Run directory name (for the messages).

    Returns:
        One message per difference.
    """
    problems = []
    for column, key in NUMERIC_KEYS.items():
        if key in config and float(config[key]) != float(row[column]):
            problems.append(f"{run}: {column} {row[column]} != {key} {config[key]}")
    for column, key in CONFIG_KEYS.items():
        names = OPERATOR_NAMES[column][row[column].split(":")[0]]
        if key in config and config[key] not in names:
            problems.append(f"{run}: {column} {row[column]} != {key} {config[key]}")
    return problems


def finished_rows(experiment: VariantExperiment) -> list[tuple[dict, Path, str | None]]:
    """Job rows of the finished runs in the result directory.

    The configuration of every run is taken from its variant and checked against ``<run>.json``;
    differences are reported, prompt and seed of the configuration file win.

    Args:
        experiment: The experiment.

    Returns:
        ``(row, run directory, Slurm job id)`` per run directory that matches a variant.
    """
    rows = []
    for run_dir in sorted(p for p in experiment.result_dir.iterdir() if p.is_dir() and p.name != "clip_scores"):
        transfer = TRANSFER_PATTERN.match(run_dir.name) if experiment.transfer_source_dir else None
        match = None if transfer else RUN_PATTERN.match(run_dir.name)
        corrected = transfer and experiment.source_corrections.get(f"{transfer['source']}_Seed_{transfer['label']}")
        source = prompt_name(corrected) if corrected else transfer and transfer["source"]
        name = f"{source}_to_{transfer['target']}" if transfer else match and match["name"]
        variant = experiment.find_variant(name) if name else None
        if variant is None:
            print(f"[WARN] {run_dir.name}: no matching variant, skipped")
            continue
        label = (transfer or match)["label"]
        config_path = run_dir / f"{run_dir.name}.json"
        config = json.loads(config_path.read_text()) if config_path.exists() else {}
        seed = str(config.get("random_seed", SEED_SETS[variant.seeds].get(label, "")))
        row = job_row(variant, label, seed, note="registered run")
        row["run_name"] = run_dir.name
        if corrected:
            row["note"] += (f"; the directory is named after {transfer['source'].replace('_', ' ')!r}, but the "
                            f"rendered noise is that of {corrected!r} seed {label}")
        if config.get("prompt") and config["prompt"] != row["prompt"]:
            print(f"[WARN] {run_dir.name}: prompt {row['prompt']!r} != {config['prompt']!r} (configuration file wins)")
            row["prompt"] = config["prompt"]
        if seed != SEED_SETS[variant.seeds].get(label):
            row["note"] += f"; seed differs from the seed set of label {label}"
        for problem in check_config(row, config, run_dir.name):
            print(f"[WARN] {problem}")
        rows.append((row, run_dir, match["job"] if match else None))
    return rows


def register_runs(experiment: VariantExperiment, db: Path, append: bool) -> None:
    """Write the finished runs of the result directory into a job database with status ``DONE``.

    Args:
        experiment: The experiment.
        db: Job database.
        append: Add to an existing database.
    """
    if db.exists() and not append:
        raise SystemExit(f"{db} already exists. Choose another --db or use --append.")
    queue = experiment.job_queue(db)
    queue.create_table()
    rows = finished_rows(experiment)
    first_id = queue.last_id() + 1
    queue.add_jobs([row for row, _, _ in rows])
    for job_id, (_, run_dir, slurm_id) in enumerate(rows, start=first_id):
        csv = run_dir / f"{run_dir.name}.csv"
        finished = datetime.fromtimestamp(csv.stat().st_mtime, timezone.utc).replace(tzinfo=None)
        queue.register_finished(job_id, str(run_dir), slurm_id, finished.isoformat(timespec="seconds"))
    print(f"{len(rows)} runs registered in {db}")


def create_jobs(experiment: VariantExperiment, db: Path, variants: list[str] | None, seeds: list[str] | None,
                append: bool) -> None:
    """Create open jobs for the variants of an experiment.

    Args:
        experiment: The experiment.
        db: Job database.
        variants: Only these variant names (default: all variants with ``rerun``).
        seeds: Only these seed labels (default: the seed set of every variant).
        append: Add to an existing database.
    """
    if db.exists() and not append:
        raise SystemExit(f"{db} already exists. Choose another --db or use --append.")
    jobs = [job_row(variant, label)
            for variant in experiment.variants
            if (variants is None and variant.rerun) or (variants is not None and variant.name in variants)
            for label in SEED_SETS[variant.seeds] if seeds is None or label in seeds]
    queue = experiment.job_queue(db)
    queue.create_table()
    print(f"{queue.add_jobs(jobs)} jobs written to {db}")
    print(f"Run a job with: {experiment.name.upper()}_DB_FILE={db} python -m experiments.{experiment.name}.worker <job id>")


def create_jobs_cli(experiment: VariantExperiment, description: str) -> None:
    """Command line entry point of ``experiments.<name>.create_jobs``.

    Args:
        experiment: The experiment.
        description: Help text (the module docstring).
    """
    parser = argparse.ArgumentParser(description=description, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--db", type=Path, default=experiment.db_file,
                        help=f"SQLite job database (default: {experiment.name.upper()}_DB_FILE)")
    parser.add_argument("--register", action="store_true",
                        help="list the finished runs of the result directory as DONE instead of creating open jobs")
    parser.add_argument("--variants", nargs="+", default=None, help="only these variants")
    parser.add_argument("--seeds", nargs="+", default=None, help="only these seed labels")
    parser.add_argument("--append", action="store_true", help="add jobs to an existing database")
    args = parser.parse_args()
    if args.register:
        register_runs(experiment, args.db, args.append)
    else:
        create_jobs(experiment, args.db, args.variants, args.seeds, args.append)


# --- worker -------------------------------------------------------------------------------------------------


def build_crossover(spec: str):
    """Create the crossover operator of a specification.

    Args:
        spec: ``uniform``, ``uniform_pixel``, ``n_point:<n>`` or ``arithmetic``.

    Returns:
        The crossover function.

    Raises:
        ValueError: For an unknown specification.
    """
    from evolutionary_creativity.operators import (ArithmeticCrossover, NPointCrossover, UniformCrossover,
                                                   UniformPixelCrossover)
    from experiments.common.settings import DEVICE

    kind, *args = spec.split(":")
    if kind == "uniform":
        return UniformCrossover(device=DEVICE)
    if kind == "uniform_pixel":
        return UniformPixelCrossover()
    if kind == "n_point":
        return NPointCrossover(int(args[0]))
    if kind == "arithmetic":
        return ArithmeticCrossover()
    raise ValueError(f"Unknown crossover {spec!r}")


def build_mutator(spec: str):
    """Create the mutation operator of a specification.

    Args:
        spec: ``gaussian:<fraction>:<strength>`` or ``uniform:<rate>``.

    Returns:
        The mutation function.

    Raises:
        ValueError: For an unknown specification.
    """
    from evolutionary_creativity.operators import UniformGaussianMutator, UniformMutator

    kind, *args = spec.split(":")
    if kind == "gaussian":
        return UniformGaussianMutator(mutation_rate=float(args[0]), mutation_strength=float(args[1]))
    if kind == "uniform":
        return UniformMutator(mutation_rate=float(args[0]))
    raise ValueError(f"Unknown mutator {spec!r}")


def build_fitness(spec: str, blip2):
    """Create the fitness evaluator of a specification.

    Args:
        spec: ``mmd``, ``mmd_archive:<size>:<threshold>`` or ``knn:<k>``.
        blip2: BLIP-2 embedding model.

    Returns:
        The fitness evaluator.

    Raises:
        ValueError: For an unknown specification.
    """
    from evolutionary_creativity.evaluators import KNearestNeighborEvaluator, MaxMeanDivergenceEvaluator
    from experiments.common.settings import DEVICE

    kind, *args = spec.split(":")
    if kind == "mmd":
        return MaxMeanDivergenceEvaluator(blip2, DEVICE)
    if kind == "mmd_archive":
        return MaxMeanDivergenceEvaluator(blip2, DEVICE, archive_size=int(args[0]), archive_threshold=float(args[1]))
    if kind == "knn":
        return KNearestNeighborEvaluator(blip2, DEVICE, n_neighbors=int(args[0]))
    raise ValueError(f"Unknown fitness {spec!r}")


def transfer_populations(job: sqlite3.Row, source_dir: Path, dtype) -> list:
    """Load the noise of the transferred generations of the source run.

    Args:
        job: Transfer job.
        source_dir: Directory with the runs of the source prompt.
        dtype: Noise dtype of the text-to-image model.

    Returns:
        ``(generation, candidates)`` for every generation of :data:`TRANSFER_GENERATIONS`; candidate ids are
        the ids of the source run.

    Raises:
        FileNotFoundError: If no source run exists.
    """
    import torch

    from evolutionary_creativity.candidate import Candidate
    from experiments.common.settings import DEVICE

    pattern = f"{prompt_name(job['source_prompt'])}_Seed_{job['seed_label']}_*"
    sources = sorted(source_dir.glob(pattern, case_sensitive=False))
    if not sources:
        raise FileNotFoundError(f"No source run {pattern} in {source_dir}")
    archive_path = next(sources[0].glob("*.zip"))
    logger.info("Transfer source: %s", archive_path)
    populations = []
    with zipfile.ZipFile(archive_path) as archive:
        for generation in TRANSFER_GENERATIONS:
            candidates = []
            for name in sorted(n for n in archive.namelist()
                               if n.startswith("/noise/") and n.rsplit("/", 1)[-1].startswith(f"g{generation}_")):
                noise = torch.load(io.BytesIO(archive.read(name)), map_location=DEVICE).to(dtype)
                candidate_id = re.search(r"idn_(\d+)", name).group(1)
                candidates.append(Candidate(id=candidate_id, initial_noise=noise, start_generation=generation,
                                            end_generation=generation))
            populations.append((generation, candidates))
    return populations


def run_job(experiment: VariantExperiment, job: sqlite3.Row) -> str:
    """Run (or, for transfer jobs, evaluate) one job.

    The recorded metrics are those of all current experiments: distance to the mean of generation 0
    (``model_LocalMaxMeanDivergence``), CLIP score (``model_CLIPScore``) and diversity within the generation
    (``model_DiversityWithinPopulation``), all in BLIP-2 space except the CLIP score.

    Args:
        experiment: The experiment.
        job: Row of the job database.

    Returns:
        Result directory of the run (or of the experiment, if the run had already finished).
    """
    import torch

    from evolutionary_creativity.models.embeddings import Blip2EmbeddingModel, ClipEmbeddingModel
    from evolutionary_creativity.models.text_to_image.sdxl import StableDiffusionXLModel
    from evolutionary_creativity.noise_factory import NoiseFactory
    from evolutionary_creativity.operators import TournamentSelector
    from evolutionary_creativity.pipeline import GeneticAlgorithmPipeline
    from experiments.common.ga import TOURNAMENT_SIZE, set_all_seeds, standard_evaluators, success_marker
    from experiments.common.settings import DEVICE, HF_CACHE

    directory = experiment.result_dir
    directory.mkdir(parents=True, exist_ok=True)
    marker = success_marker(directory, job["run_name"])
    if marker.exists():
        logger.info("Job %s already finished (%s), skipping", job["id"], marker)
        return str(directory)
    logger.info("Job %s | %s", job["id"], dict(job))

    set_all_seeds(int(job["seed"]))
    sdxl = StableDiffusionXLModel(DEVICE, HF_CACHE, num_inference_steps=int(job["inference_steps"]),
                                  torch_dtype=torch.float16)
    clip = ClipEmbeddingModel(DEVICE, HF_CACHE)
    blip2 = Blip2EmbeddingModel(DEVICE, HF_CACHE)
    _, metrics = standard_evaluators(blip2, clip, DEVICE)

    transfer = job["source_prompt"] is not None
    run_id = job["run_name"] if transfer else f"{job['run_name']}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    pipe = GeneticAlgorithmPipeline(
        generative_model=sdxl,
        prompt=job["prompt"],
        crossover_operation=build_crossover(job["crossover"]),
        selector=TournamentSelector(tournament_size=TOURNAMENT_SIZE),
        mutator=build_mutator(job["mutator"]),
        noise_factory=NoiseFactory(),
        fitness_evaluator=[build_fitness(job["fitness"], blip2)],
        model_evaluator=metrics,
        experiment_id=run_id,
        num_generations=int(job["num_generations"]),
        population_size=int(job["population_size"]),
        mutation_rate=float(job["mutation_rate"]),
        crossover_rate=float(job["crossover_rate"]),
        elite_size=int(job["elite_size"]),
        batch_size=int(job["batch_size"]),
        result_path=str(directory),
    )
    pipe.save_config()
    if transfer:
        pipe.evaluate_populations(transfer_populations(job, experiment.transfer_source_dir, sdxl.noise_config().dtype))
    else:
        pipe.run()
    marker.write_text("ok\n", encoding="utf-8")
    return str(pipe.result_path)


def worker_cli(experiment: VariantExperiment) -> None:
    """Command line entry point of ``experiments.<name>.worker``.

    Args:
        experiment: The experiment.
    """
    from experiments.common.ga import run_queued_job

    run_queued_job(experiment.job_queue(), lambda job: run_job(experiment, job),
                   f"Run one job of the experiment {experiment.name}.")
