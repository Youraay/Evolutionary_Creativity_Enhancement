"""Parent selection operators."""
import random
from abc import ABC, abstractmethod

from evolutionary_creativity.candidate import Candidate


class SelectorFunction(ABC):
    """Chooses a parent from the current population."""

    @abstractmethod
    def select(self, contenders: list[Candidate]) -> Candidate:
        """Pick one parent.

        Args:
            contenders: Evaluated population (every candidate has a fitness).

        Returns:
            The selected candidate.
        """


class TournamentSelector(SelectorFunction):
    """Tournament selection: the fittest of ``tournament_size`` randomly drawn candidates wins."""

    name = "TournamentSelector"

    def __init__(self, tournament_size: int) -> None:
        """
        Args:
            tournament_size: Number of contenders per tournament, at least 2.

        Raises:
            AssertionError: If ``tournament_size`` is smaller than 2.
        """
        assert tournament_size > 1, "Tournament is only possible with two or more contenders"
        self.tournament_size = tournament_size

    def select(self, contenders: list[Candidate]) -> Candidate:
        """Draw ``tournament_size`` distinct candidates with Python's ``random`` and return the fittest.

        Args:
            contenders: Evaluated population.

        Returns:
            The candidate with the highest fitness among the drawn ones.
        """
        drawn = random.sample(contenders, self.tournament_size)
        return max(drawn, key=lambda candidate: candidate.fitness)
