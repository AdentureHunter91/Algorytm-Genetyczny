from __future__ import annotations

import random
from typing import List, Sequence, Tuple


# ============================
#   SELECTION
# ============================

def tournament_selection(population: List[List[int]], fitnesses: List[float], k: int, rng: random.Random) -> List[int]:
    """ Selekcja turniejowa – wybiera najlepszego z losowych k osobników """
    candidates = rng.sample(list(zip(population, fitnesses)), k)
    best = min(candidates, key=lambda item: item[1])  # wybieramy najmniejszy dystans = najlepszy
    return best[0]


def roulette_wheel_selection(population: List[List[int]], fitnesses: List[float], rng: random.Random) -> List[int]:
    """ Selekcja ruletki – im lepszy osobnik tym większa szansa na wybór """
    max_fit = max(fitnesses)
    adjusted = [max_fit - f + 1e-9 for f in fitnesses]  # odwracamy bo w TSP im mniej tym lepiej
    total = sum(adjusted)
    pick = rng.random() * total

    current = 0.0
    for individual, weight in zip(population, adjusted):
        current += weight
        if current >= pick:
            return individual
    return population[-1]


# Helper do pobierania segmentu z marginesami
def get_segment_range(size: int, min_seg: int = 3) -> Tuple[int, int]:
    """
    Zwraca zakres [a, b) z zagwarantowanym:
    - min. 1 miejsce z lewej i prawej
    - minimalna długość odcinka min_seg
    """
    a = random.randint(1, size - min_seg - 1)  # a > 0
    b = random.randint(a + min_seg, size - 1)  # b < size
    return a, b


# ============================
#   CROSSOVER OX (z marginesami)
# ============================

def order_crossover(parent1: Sequence[int], parent2: Sequence[int], rng: random.Random) -> Tuple[List[int], List[int]]:
    """ OX – Order Crossover z minimalnym rozmiarem segmentu """
    size = len(parent1)
    a, b = get_segment_range(size, min_seg=3)

    def ox(p1, p2):
        child = [None] * size
        child[a:b] = p1[a:b]  # kopiujemy środkowy fragment
        pos = b
        for gene in p2:  # wypełniamy kolejnymi genami w kolejności
            if gene not in child:
                if pos >= size:
                    pos = 0
                child[pos] = gene
                pos += 1
        return child

    return ox(parent1, parent2), ox(parent2, parent1)


# ============================
#   PMX – Partially Mapped Crossover (z marginesami)
# ============================

def pmx_crossover(parent1: Sequence[int], parent2: Sequence[int], rng: random.Random) -> Tuple[List[int], List[int]]:
    size = len(parent1)
    a, b = get_segment_range(size, min_seg=3)

    def pmx(p1, p2):
        child = [None] * size
        child[a:b] = p1[a:b]

        mapping = {p2[i]: p1[i] for i in range(a, b)}  # mapowanie konfliktów między rodzicami

        for i in range(size):
            if child[i] is None:
                candidate = p2[i]
                while candidate in mapping:  # rozwiązywanie konfliktów
                    candidate = mapping[candidate]
                child[i] = candidate
        return child

    return pmx(parent1, parent2), pmx(parent2, parent1)


# ============================
#   MUTATIONS
# ============================

def swap_mutation(individual: List[int], rng: random.Random) -> List[int]:
    """Swap mutation – zamienia dwa losowe geny (bez duplikatów, zawsze legalna permutacja)"""
    a, b = rng.sample(range(len(individual)), 2)
    mutant = individual.copy()
    mutant[a], mutant[b] = mutant[b], mutant[a]
    return mutant


def inversion_mutation(individual: List[int], rng: random.Random, gen: int = 0, generations: int = 1000) -> List[int]:
    """
    Inversion mutation z *adaptive mutation rate* i marginesami segmentu.
    Mutacja na początku duża – zmniejsza się wraz z ewolucją.
    """
    size = len(individual)
    min_seg = 3

    # ADAPTIVE MUTATION
    base_mut = 0.20  # początkowa siła mutacji (wysoka)
    decay = gen / generations  # im później, tym mniejsza mutacja
    mut_rate = base_mut * (1 - decay) + 0.05  # nigdy nie spada do zera

    if random.random() < mut_rate:
        a, b = get_segment_range(size, min_seg)
        mutant = individual.copy()
        mutant[a:b] = reversed(mutant[a:b])
        return mutant

    return individual.copy()


# mapy operatorów – dzięki temu wystarczy przekazać string w GA
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
