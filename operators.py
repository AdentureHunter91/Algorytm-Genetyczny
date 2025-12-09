from __future__ import annotations
import random
from typing import List, Sequence, Tuple


# ===============================
#   SELECTION
# ===============================

def tournament_selection(population: List[List[int]], fitnesses: List[float], k: int, rng: random.Random) -> List[int]:
    """Selekcja turniejowa – wybiera najlepszego z losowo pobranych k osobników."""
    candidates = rng.sample(list(zip(population, fitnesses)), k)
    best = min(candidates, key=lambda x: x[1])
    return best[0]


def roulette_wheel_selection(population: List[List[int]], fitnesses: List[float], rng: random.Random) -> List[int]:
    """Ruletka – im lepszy (mniejszy dystans), tym większa szansa wyboru."""
    max_fit = max(fitnesses)
    weights = [max_fit - f + 1e-9 for f in fitnesses]
    total = sum(weights)
    pick = rng.random() * total
    curr = 0.0
    for ind, w in zip(population, weights):
        curr += w
        if curr >= pick:
            return ind
    return population[-1]


# ===============================
#  HELPER – zakres z marginesem
# ===============================

def get_segment_range(size: int, min_seg: int = 3) -> Tuple[int, int]:
    """
    Losuje odcinek [a,b) z gwarancją:
      – min. segmentu = min_seg
      – nie zaczyna na 0
      – nie kończy na size-1
    """
    a = random.randint(1, size - min_seg - 1)
    b = random.randint(a + min_seg, size - 1)
    return a, b


# ===============================
#   CROSSOVER – OX (z marginesami)
# ===============================

def order_crossover(p1: Sequence[int], p2: Sequence[int], rng: random.Random) -> Tuple[List[int], List[int]]:
    size = len(p1)
    a, b = get_segment_range(size, min_seg=3)

    def ox(A, B):
        child = [None] * size
        child[a:b] = A[a:b]  # kopiujemy środek

        pos = b
        for gene in B:
            if gene not in child:
                if pos >= size: pos = 0
                child[pos] = gene
                pos += 1
        return child

    return ox(p1, p2), ox(p2, p1)


# ===============================
#   CROSSOVER – PMX klasyczny
# ===============================

def pmx_crossover(p1: Sequence[int], p2: Sequence[int], rng: random.Random) -> Tuple[List[int], List[int]]:
    size = len(p1)
    a, b = get_segment_range(size, min_seg=3)

    def pmx(A, B):
        child = [None] * size
        child[a:b] = A[a:b]
        mapping = {B[i]: A[i] for i in range(a, b)}

        for i in range(size):
            if child[i] is None:
                gene = B[i]
                visited = set()

                # 🔒 zabezpieczenie przed zapętlaniem w cyklicznym mapowaniu
                while gene in mapping and gene not in visited:
                    visited.add(gene)
                    gene = mapping[gene]

                # jeśli cykl nie pozwolił na wyjście poza mapowanie, wybierz pierwszy wolny gen
                if gene in mapping:
                    for candidate in B:
                        if candidate not in child:
                            gene = candidate
                            break

                child[i] = gene
        return child

    return pmx(p1, p2), pmx(p2, p1)


# ===============================
#   CROSSOVER – PMX STRICT (pro)
# ===============================
def pmx_strict(p1, p2, rng):
    size = len(p1)
    start, end = get_segment_range(size, min_seg=3)

    def create_child(A, B):
        child = [-1] * size
        child[start:end] = A[start:end]

        MAX_HOPS = size * 3  # 🔥 zabezpiecza przed nieskończoną pętlą

        for i in list(range(start)) + list(range(end, size)):
            val = B[i]
            hops = 0

            # 🔥 poprawione mapowanie — bez ryzyka .index() crash
            while val in child[start:end] and hops < MAX_HOPS:
                try:
                    idx = A.index(val)       # zamiast A[start:end] → pełne mapowanie
                    val = B[idx]
                except ValueError:
                    break                     # brak wartości → wychodzimy
                hops += 1

            # fallback gdy mapping zapętlił się
            if val in child:
                for v in B:
                    if v not in child:
                        val = v
                        break

            child[i] = val

        return child

    return create_child(p1, p2), create_child(p2, p1)

# ===============================
#   MUTATION – SWAP
# ===============================

def swap_mutation(ind: List[int], rng: random.Random) -> List[int]:
    a, b = rng.sample(range(len(ind)), 2)
    c = ind.copy()
    c[a], c[b] = c[b], c[a]
    return c


# ===============================
#   MUTATION – INVERSION + adaptive
# ===============================

def inversion_mutation(ind: List[int], rng: random.Random, gen: int = 0, generations: int = 1000) -> List[int]:
    size = len(ind)

    # siła spada wraz z generacjami
    base = 0.25                         # agresywna na starcie
    decay = gen / generations
    mut_rate = base * (1 - decay) + 0.05  # nigdy nie spada do zera

    if random.random() < mut_rate:
        a, b = get_segment_range(size, min_seg=3)
        c = ind.copy()
        c[a:b] = reversed(c[a:b])
        return c
    return ind.copy()


# ===============================
#  mapping słownikowy → GA wybiera stringiem
# ===============================

SELECTION_OPERATORS = {
    "tournament": tournament_selection,
    "roulette": roulette_wheel_selection,
}

CROSSOVER_OPERATORS = {
    "ox": order_crossover,
    "pmx": pmx_crossover,
    "pmx_strict": pmx_strict,      # najlepszy do użycia!
}

MUTATION_OPERATORS = {
    "swap": swap_mutation,
    "inversion": inversion_mutation,
}
