"""Genetic algorithm over the initial noise of a text-to-image model."""
import json
import logging
import random
import time
import zipfile
from collections.abc import Iterator
from datetime import datetime
from pathlib import Path

import pandas as pd
import torch

from evolutionary_creativity.candidate import Candidate
from evolutionary_creativity.evaluators.base import BaseImageEvaluator, EvaluationResult
from evolutionary_creativity.models.text_to_image.base import GenerativeModel
from evolutionary_creativity.noise_factory import NoiseFactory
from evolutionary_creativity.operators import CrossoverFunction, MutationFunction, SelectorFunction

logger = logging.getLogger(__name__)


def _log_peak_memory(tag: str) -> None:
    """Log the peak CUDA memory since the last reset (debug level).

    Args:
        tag: Label of the log line.
    """
    if torch.cuda.is_available():
        peak = torch.cuda.max_memory_allocated(torch.cuda.current_device()) / 1024**3
        logger.debug("[GPU] %s | peak allocated=%.2f GB", tag, peak)


class GeneticAlgorithmPipeline:
    """Evolves a population of initial noise tensors towards novel images.

    One run consists of an initial generation and ``num_generations`` evolution steps. In every
    generation all candidates are rendered, scored by the fitness evaluators (used for selection) and by
    the model evaluators (analysis only), and written to the result directory:

    * ``<run>/<run>.json``: configuration (:meth:`save_config`),
    * ``<run>/<run>.csv``: one row per candidate and generation with all scores,
    * ``<run>/<run>.zip``: images, embeddings and initial noise of every candidate.

    Evolution keeps the ``elite_size`` fittest candidates and fills the population with children: with
    probability ``crossover_rate`` a child is the crossover of two selected parents, otherwise a copy of
    one parent; each child is mutated with probability ``mutation_rate``.
    """

    def __init__(self, generative_model: GenerativeModel, crossover_operation: CrossoverFunction,
                 noise_factory: NoiseFactory, mutator: MutationFunction, selector: SelectorFunction,
                 fitness_evaluator: list[BaseImageEvaluator], model_evaluator: list[BaseImageEvaluator], prompt: str,
                 experiment_id: str, num_generations: int, population_size: int, mutation_rate: float,
                 crossover_rate: float, elite_size: int = 0, batch_size: int = 1,
                 result_path: str = "results") -> None:
        """
        Args:
            generative_model: Text-to-image model that renders the candidates.
            crossover_operation: Crossover operator.
            noise_factory: Creates the initial population and the ids of new candidates.
            mutator: Mutation operator.
            selector: Parent selection operator.
            fitness_evaluator: Evaluators whose mean score is the fitness. They update their reference every
                generation.
            model_evaluator: Evaluators recorded for analysis only. They set their reference in generation 0.
            prompt: Prompt used for generation (and passed to the evaluators).
            experiment_id: Name of the run directory and of its files.
            num_generations: Number of evolution steps after the initial generation.
            population_size: Number of candidates per generation.
            mutation_rate: Probability that a child is mutated.
            crossover_rate: Probability that a child is created by crossover instead of copying a parent.
            elite_size: Number of fittest candidates copied unchanged into the next generation.
            batch_size: Number of images generated per call of the text-to-image model.
            result_path: Directory in which the run directory is created.
        """
        self.generative_model = generative_model
        self.crossover_operation = crossover_operation
        self.noise_factory = noise_factory
        self.mutator = mutator
        self.selection_function = selector
        self.fitness_evaluator = fitness_evaluator
        self.model_evaluator = model_evaluator
        self.prompt = prompt
        self.num_generations = num_generations
        self.population_size = population_size
        self.mutation_rate = mutation_rate
        self.crossover_rate = crossover_rate
        self.elite_size = elite_size
        self.batch_size = batch_size
        self.population: list[Candidate] = []
        self.generations_done = 0

        self.name = experiment_id
        self.result_path = Path(result_path) / self.name
        self.zip_path = self.result_path / f"{self.name}.zip"
        self.state_path = self.result_path / f"{self.name}.csv"
        self.config_path = self.result_path / f"{self.name}.json"

        # The CSV is appended to; leftovers of an aborted run would otherwise mix with the new run.
        for stale in (self.zip_path, self.state_path):
            if stale.exists():
                logger.warning("Removing result file of an earlier run: %s", stale)
                stale.unlink()

    def save_config(self) -> None:
        """Write the run configuration (operators, rates, seed, paths) to ``<run>.json``."""
        config = {
            "num_generations": self.num_generations,
            "population_size": self.population_size,
            "batch_size": self.batch_size,
            "prompt": self.prompt,
            "generative_model": type(self.generative_model).__name__,
            "selector": type(self.selection_function).__name__,
            "mutator": type(self.mutator).__name__,
            "crossover_function": type(self.crossover_operation).__name__,
            "initial_mutation_rate": self.mutation_rate,
            "crossover_rate": self.crossover_rate,
            "elitism_count": self.elite_size,
            "fitness_evaluators": [type(e).__name__ for e in self.fitness_evaluator],
            "model_evaluators": [type(e).__name__ for e in self.model_evaluator],
            "random_seed": torch.initial_seed(),
            "timestamp": datetime.now().strftime("%Y%m%d_%H%M%S"),
            "result_path": str(self.result_path),
            "stat_path": str(self.state_path),
        }
        self.config_path.parent.mkdir(parents=True, exist_ok=True)
        self.config_path.write_text(json.dumps(config, indent=4))

    def run(self) -> list[Candidate]:
        """Run the initial generation and all evolution steps, saving every generation.

        Returns:
            The final population.
        """
        logger.info("Generating the initial population")
        self._initial_population()
        self._save_generation()
        self._save_states()
        while self.generations_done < self.num_generations:
            logger.info("Next generation: %d", self.generations_done + 1)
            steps = (("evolve", self._evolve), ("one_generation", self._evaluate_population),
                     ("save_generation", self._save_generation), ("save_states", self._save_states))
            for label, step in steps:
                start = time.time()
                step()
                logger.info("%s: %.1fs", label, time.time() - start)
        return self.population

    def evaluate_populations(self, populations: list[tuple[int, list[Candidate]]]) -> None:
        """Render, evaluate and save given populations without evolution (e.g. noise transferred from another run).

        Every population is stored as the given generation; analysis evaluators set their reference with
        the population of generation 0.

        Args:
            populations: ``(generation, candidates)`` pairs in the order of evaluation.
        """
        for generation, population in populations:
            self.generations_done = generation
            self.population = population
            self._evaluate_population()
            self._save_generation()
            self._save_states()

    def _batches(self, items: list) -> Iterator[list]:
        """Split ``items`` into consecutive chunks of ``batch_size``."""
        for i in range(0, len(items), self.batch_size):
            yield items[i:i + self.batch_size]

    def _initial_population(self) -> None:
        """Draw the initial population and evaluate it as generation 0."""
        self.population = self.noise_factory.create_population(self.generative_model.noise_config(),
                                                               self.population_size)
        noise = self.population[0].initial_noise
        logger.info("Population size: %d | noise shape: %s | dtype: %s", len(self.population), tuple(noise.shape),
                    noise.dtype)
        self._evaluate_population()
        for candidate in self.population:
            candidate.start_generation = 0
            candidate.end_generation = 0

    def _evaluate_population(self) -> None:
        """Render all candidates, run all evaluators and compute the fitness."""
        images = []
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
        for batch in self._batches([candidate.initial_noise for candidate in self.population]):
            images.extend(self.generative_model.generate_batch(batch, self.prompt))
        _log_peak_memory("after image generation")

        for candidate, image in zip(self.population, images):
            candidate.pil_image = image
            candidate.evaluation_scores = []
            candidate.global_scores = []
            candidate.embeddings = {}
            candidate.metadata = {}

        initial = self.generations_done == 0
        for evaluator in self.model_evaluator:
            self._collect(evaluator.evaluate(images, self.prompt, update_reference=initial), "global_scores")
            _log_peak_memory(f"after {evaluator.name}")
        for evaluator in self.fitness_evaluator:
            self._collect(evaluator.evaluate(images, self.prompt, update_reference=True), "evaluation_scores")

        for candidate in self.population:
            candidate.calculate_fitness()

    def _collect(self, results: list[EvaluationResult], target: str) -> None:
        """Attach evaluator results to the candidates.

        Args:
            results: One result per candidate, in population order.
            target: ``"global_scores"`` (analysis) or ``"evaluation_scores"`` (fitness).
        """
        for candidate, result in zip(self.population, results):
            getattr(candidate, target).append({"name": result["name"], "score": result["score"]})
            candidate.embeddings.update(result.get("embeddings", {}))
            if "metadata" in result:
                candidate.metadata[result["name"]] = result["metadata"]

    def _evolve(self) -> None:
        """Replace the population by the elite and newly created children.

        The order of the random draws (selection, crossover decision, crossover mask, mutation) is part
        of the reproducibility of a run and must not change.
        """
        self.generations_done += 1
        self.population = sorted(self.population, key=lambda candidate: candidate.fitness, reverse=True)
        new_population = self.population[:self.elite_size]
        for candidate in new_population:
            candidate.end_generation = self.generations_done

        actions = []
        while len(new_population) < self.population_size:
            parent_1 = self.selection_function.select(self.population)
            parent_2 = self.selection_function.select(self.population)

            if random.random() > self.crossover_rate:
                child = self.noise_factory.from_tensor(parent_1.initial_noise.clone())
                # The copy gets a second fresh id (the counter advances twice); this keeps the candidate
                # ids, and thus the file names, identical to earlier runs.
                child.id = self.noise_factory.create_id()
                child.parent_1, child.parent_2 = parent_1.id, ""
                child.start_generation = parent_1.start_generation
                child.crossover = False
                action = "Parent"
            else:
                child = self.noise_factory.from_tensor(
                    self.crossover_operation.crossover(parent_1.initial_noise, parent_2.initial_noise))
                child.parent_1, child.parent_2 = parent_1.id, parent_2.id
                child.start_generation = self.generations_done
                child.crossover = True
                action = "Cross"
            child.end_generation = self.generations_done
            child.mutate = False

            if random.random() < self.mutation_rate:
                child.initial_noise = self.mutator.mutate(child.initial_noise)
                child.mutate = True
                action += "+Mut"
            new_population.append(child)
            actions.append(action)

        self.population = new_population
        logger.info("Gen %d: %s", self.generations_done, "; ".join(actions))

    def _save_generation(self) -> None:
        """Append images, embeddings and noise of the current population to the result archive."""
        self.result_path.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(self.zip_path, "a", compression=zipfile.ZIP_DEFLATED) as archive:
            for candidate in self.population:
                candidate.add_to_zip(archive, folder_prefix="")

    def _save_states(self) -> None:
        """Append one CSV row per candidate (lineage, fitness and all scores) to ``<run>.csv``."""
        rows = []
        for candidate in self.population:
            row = {
                "generation": self.generations_done,
                "candidate_id": candidate.id,
                "start_gen": candidate.start_generation,
                "end_gen": candidate.end_generation,
                "parent_1": candidate.parent_1,
                "parent_2": candidate.parent_2,
                "fitness": candidate.fitness,
                "caption": candidate.caption,
                "file_name": candidate.filename,
                "metadata": json.dumps(candidate.metadata) if candidate.metadata else "{}",
            }
            row.update({f"fitness_{s['name']}": s["score"] for s in candidate.evaluation_scores})
            row.update({f"model_{s['name']}": s["score"] for s in candidate.global_scores})
            rows.append(row)

        self.state_path.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(rows).to_csv(self.state_path, mode="a", header=not self.state_path.exists(), index=False)
