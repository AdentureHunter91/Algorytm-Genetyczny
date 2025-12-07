from __future__ import annotations

import random
from typing import List, Sequence, Tuple


def tournament_selection(population: List[List[int]], fitnesses: List[float], k: int, rng: random.Random) -> List[int]:
    candidates = rng.sample(list(zip(population, fitnesses)), k)
    best = min(candidates, key=lambda item: item[1])
    return best[0]


def roulette_wheel_selection(population: List[List[int]], fitnesses: List[float], rng: random.Random) -> List[int]:
    max_fit = max(fitnesses)
    adjusted = [max_fit - f + 1e-9 for f in fitnesses]
    total = sum(adjusted)
    pick = rng.random() * total
    current = 0.0
    for individual, weight in zip(population, adjusted):
        current += weight
        if current >= pick:
            return individual
    return population[-1]


def order_crossover(parent1: Sequence[int], parent2: Sequence[int], rng: random.Random) -> Tuple[List[int], List[int]]:
    size = len(parent1)
    a, b = sorted(rng.sample(range(size), 2))
    def ox(p1: Sequence[int], p2: Sequence[int]) -> List[int]:
        child = [None] * size  # type: ignore
        child[a:b] = p1[a:b]
        pos = b
        for gene in p2:
            if gene not in child:
                if pos >= size:
                    pos = 0
                child[pos] = gene
                pos += 1
        return child  # type: ignore
    return ox(parent1, parent2), ox(parent2, parent1)


def pmx_crossover(parent1: Sequence[int], parent2: Sequence[int], rng: random.Random) -> Tuple[List[int], List[int]]:
    size = len(parent1)
    a, b = sorted(rng.sample(range(size), 2))
    def pmx(p1: Sequence[int], p2: Sequence[int]) -> List[int]:
        child = [None] * size  # type: ignore
        child[a:b] = p1[a:b]
        mapping = {p2[i]: p1[i] for i in range(a, b)}
        for i in range(size):
            if child[i] is None:
                candidate = p2[i]
                while candidate in mapping:
                    candidate = mapping[candidate]
                child[i] = candidate
        return child  # type: ignore
    return pmx(parent1, parent2), pmx(parent2, parent1)


def swap_mutation(individual: List[int], rng: random.Random) -> List[int]:
    a, b = rng.sample(range(len(individual)), 2)
    mutant = individual.copy()
    mutant[a], mutant[b] = mutant[b], mutant[a]
    return mutant


def inversion_mutation(individual: List[int], rng: random.Random) -> List[int]:
    a, b = sorted(rng.sample(range(len(individual)), 2))
    mutant = individual.copy()
    mutant[a:b] = reversed(mutant[a:b])
    return mutant


SELECTION_OPERATORS = {
    "tournament": tournament_selection,
    "roulette": roulette_wheel_selection,
}

CROSSOVER_OPERATORS = {
    "ox": order_crossover,
    "pmx": pmx_crossover,
}

MUTATION_OPERATORS = {
    "swap": swap_mutation,
    "inversion": inversion_mutation,
}
