"""Evolutionary optimisation of the initial noise of text-to-image diffusion models.

A genetic algorithm evolves a population of initial noise tensors. Every candidate is turned into an
image by a frozen text-to-image model, embedded with an image encoder and scored by its distance to the
mean embedding of the current population (novelty). Selection, crossover and mutation then act directly
on the noise tensors.

Subpackages:
    evaluators  Fitness and analysis metrics computed on the generated images.
    models      Wrappers around text-to-image models and image embedding models.
    operators   Selection, crossover and mutation operators of the genetic algorithm.
"""
