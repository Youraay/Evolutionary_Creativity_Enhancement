"""A single individual of the genetic algorithm."""
import io
import zipfile
from dataclasses import dataclass, field
from typing import Any

import torch
from PIL import Image


@dataclass
class Candidate:
    """Initial noise tensor together with the image and the scores it produced.

    Attributes:
        id: Unique identifier within a run, e.g. ``"n_0042"``.
        initial_noise: Latent noise tensor of shape ``[1, C, H, W]`` that seeds the diffusion model.
        pil_image: Image generated from ``initial_noise`` in the current generation.
        fitness: Mean of all fitness scores (see :meth:`calculate_fitness`).
        evaluation_scores: Fitness scores as ``{"name": str, "score": float}`` dicts.
        global_scores: Analysis scores (not used for selection) in the same format.
        embeddings: Image embeddings by embedding name, stored in the result archive.
        metadata: Optional per-evaluator metadata, stored as JSON in the state CSV.
        caption: Always ``None``; kept so the state CSV keeps the column layout of earlier runs.
        start_generation: Generation in which this noise was created.
        end_generation: Generation in which this candidate was evaluated last.
        parent_1: Id of the first parent (empty for the initial population).
        parent_2: Id of the second parent (empty for copies and the initial population).
        crossover: Whether the candidate was created by crossover.
        mutate: Whether the candidate was mutated.
    """

    id: str
    initial_noise: torch.Tensor
    pil_image: Image.Image | None = None
    fitness: float | None = None
    evaluation_scores: list[dict] = field(default_factory=list)
    global_scores: list[dict] = field(default_factory=list)
    embeddings: dict[str, torch.Tensor] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)
    caption: str | None = None
    start_generation: int | None = None
    end_generation: int | None = None
    parent_1: str = ""
    parent_2: str = ""
    crossover: bool | None = None
    mutate: bool | None = None

    @property
    def filename(self) -> str:
        """File stem used in the result archive, e.g. ``"g3_idn_0042_f0.0812"``.

        Returns:
            ``g<end_generation>_id<id>_f<fitness>``; a missing fitness is written as ``0.0``.
        """
        fitness = 0.0 if self.fitness is None else self.fitness
        return f"g{self.end_generation}_id{self.id}_f{fitness}"

    def calculate_fitness(self) -> None:
        """Set :attr:`fitness` to the mean of all fitness scores, or ``-999.0`` if there are none."""
        if not self.evaluation_scores:
            self.fitness = -999.0
            return
        scores = [s["score"] if isinstance(s, dict) else s for s in self.evaluation_scores]
        self.fitness = sum(scores) / len(scores)

    def add_to_zip(self, archive: zipfile.ZipFile, folder_prefix: str = "") -> None:
        """Write image, embeddings and initial noise of this candidate into an open zip archive.

        Layout inside the archive: ``<prefix>/images/<filename>.JPEG``, ``<prefix>/<embedding>/<filename>.pt``
        and ``<prefix>/noise/<filename>.pt``.

        Args:
            archive: Zip archive opened for writing or appending.
            folder_prefix: Path prefix inside the archive.
        """
        if self.pil_image is not None:
            buffer = io.BytesIO()
            self.pil_image.save(buffer, format="JPEG")
            archive.writestr(f"{folder_prefix}/images/{self.filename}.JPEG", buffer.getvalue())

        for name, tensor in self.embeddings.items():
            buffer = io.BytesIO()
            torch.save(tensor, buffer)
            archive.writestr(f"{folder_prefix}/{name}/{self.filename}.pt", buffer.getvalue())

        if self.initial_noise is not None:
            buffer = io.BytesIO()
            torch.save(self.initial_noise, buffer)
            archive.writestr(f"{folder_prefix}/noise/{self.filename}.pt", buffer.getvalue())
