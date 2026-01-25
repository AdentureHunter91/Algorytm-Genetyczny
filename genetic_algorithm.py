from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Callable, List, Tuple

import numpy as np

from operators import (
    CROSSOVER_OPERATORS,
    MUTATION_OPERATORS,
    SELECTION_OPERATORS,
)
from tsp_problem import TSP
from utils import total_route_length, two_opt


@dataclass
class GAResult:
    best_route: List[int]
    best_distance: float
    history: List[float]


class GeneticAlgorithm:
    def __init__(
        self,
        problem: TSP,
        population_size: int,
        generations: int,
        mutation_prob: float,
        crossover_prob: float,
        selection_method: str = "tournament",
        crossover_method: str = "ox",
        mutation_method: str = "swap",
        elite_ratio: float = 0.05,
        tournament_k: int = 5,
        seed: int | None = None,
        two_opt_prob: float = 0.2,
    ) -> None:
        self.problem = problem
        self.population_size = population_size
        self.generations = generations
        self.mutation_prob = mutation_prob
        self.crossover_prob = crossover_prob
        self.selection_method = selection_method
        self.crossover_method = crossover_method
        self.mutation_method = mutation_method
        self.elite_ratio = elite_ratio
        self.tournament_k = tournament_k
        self.rng = random.Random(seed)
        self.distances = problem.distances
        self.two_opt_prob = two_opt_prob
        if self.distances is None:
            raise ValueError("Problem distances are not initialized")
        self.selection_fn = SELECTION_OPERATORS[selection_method]
        self.crossover_fn = CROSSOVER_OPERATORS[crossover_method]
        self.mutation_fn = MUTATION_OPERATORS[mutation_method]

    def _initial_population(self) -> List[List[int]]:
        base = list(range(self.problem.dimension))
        population = []
        for _ in range(self.population_size):
            individual = base.copy()
            self.rng.shuffle(individual)
            population.append(individual)
        return population

    def _fitness(self, individual: List[int]) -> float:
        return total_route_length(individual, self.distances)

    def _select(self, population: List[List[int]], fitnesses: List[float]) -> List[int]:
        if self.selection_method == "tournament":
            return self.selection_fn(population, fitnesses, self.tournament_k, self.rng)
        return self.selection_fn(population, fitnesses, self.rng)

    def _crossover(self, parent1: List[int], parent2: List[int]) -> Tuple[List[int], List[int]]:
        return self.crossover_fn(parent1, parent2, self.rng)

    def _mutate(self, individual: List[int], generation: int) -> List[int]:
        if self.mutation_method == "inversion":
            return self.mutation_fn(individual, self.rng, generation, self.generations)
        return self.mutation_fn(individual, self.rng)

    def _maybe_two_opt(self, individual: List[int]) -> List[int]:
        if self.two_opt_prob <= 0:
            return individual
        if self.rng.random() < self.two_opt_prob:
            return two_opt(individual, self.distances)
        return individual

    def run(self, callback: Callable[[int, float, List[int]], None] | None = None) -> GAResult:
        population = self._initial_population()
        fitnesses = [self._fitness(ind) for ind in population]
        history: List[float] = []

        elite_count = max(0, int(self.population_size * self.elite_ratio))

        for generation in range(self.generations):
            combined = list(zip(population, fitnesses))
            combined.sort(key=lambda x: x[1])
            elites = [ind for ind, _ in combined[:elite_count]]
            best_distance = combined[0][1]
            history.append(best_distance)
            if callback:
                callback(generation, best_distance, combined[0][0])

            new_population: List[List[int]] = elites.copy()
            while len(new_population) < self.population_size:
                parent1 = self._select(population, fitnesses)
                parent2 = self._select(population, fitnesses)

                if self.rng.random() < self.crossover_prob:
                    child1, child2 = self._crossover(parent1, parent2)
                else:
                    child1, child2 = parent1.copy(), parent2.copy()

                if self.rng.random() < self.mutation_prob:
                    child1 = self._mutate(child1, generation)
                if self.rng.random() < self.mutation_prob:
                    child2 = self._mutate(child2, generation)

                child1 = self._maybe_two_opt(child1)
                child2 = self._maybe_two_opt(child2)

                new_population.extend([child1, child2])

            population = new_population[: self.population_size]
            fitnesses = [self._fitness(ind) for ind in population]

        best_idx = int(np.argmin(fitnesses))
        best_route = population[best_idx]
        best_distance = fitnesses[best_idx]
        history.append(best_distance)
        return GAResult(best_route, best_distance, history)
