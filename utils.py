from __future__ import annotations

import math
import random
from typing import Iterable, List

import numpy as np


def euclidean_distance(p1: Iterable[float], p2: Iterable[float]) -> float:
    x1, y1 = p1
    x2, y2 = p2
    return math.hypot(x1 - x2, y1 - y2)


def att_distance(p1: Iterable[float], p2: Iterable[float]) -> float:
    x1, y1 = p1
    x2, y2 = p2
    rij = math.sqrt(((x1 - x2) ** 2 + (y1 - y2) ** 2) / 10.0)
    tij = int(rij)
    if tij < rij:
        return tij + 1
    return tij


def distance_matrix(coords: List[Iterable[float]], metric: str) -> np.ndarray:
    n = len(coords)
    matrix = np.zeros((n, n), dtype=float)
    distance_fn = euclidean_distance if metric == "EUC_2D" else att_distance
    for i in range(n):
        for j in range(i + 1, n):
            d = distance_fn(coords[i], coords[j])
            matrix[i, j] = matrix[j, i] = d
    return matrix


def total_route_length(route: List[int], distances: np.ndarray) -> float:
    """Liczymy odległość drogi"""
    if len(route) == 0:
        return float("inf")
    arr = np.asarray(route, dtype=int)
    return float(np.sum(distances[arr, np.roll(arr, -1)]))


def two_opt(route: List[int], distances: np.ndarray, max_iterations: int = 10) -> List[int]:
    """Lokalna poprawka trasy: 2-opt próbuje skrócić trasę przez zamianę dwóch krawędzi."""

    # 2-opt ma sens dopiero, gdy mamy przynajmniej 4 miasta (żeby dało się "odwrócić" fragment)
    if len(route) < 4:
        return route

    best = route.copy() # pracujemy na kopii, nie psujemy oryginału
    n = len(best)

    # Robimy kilka przebiegów (limit), żeby nie mielić w nieskończoność (to przyspiesza)
    for _ in range(max_iterations):
        improved = False  # flaga: czy w tej iteracji znaleźliśmy jakąkolwiek poprawę

        # i oraz k wyznaczają fragment trasy, który potencjalnie odwrócimy
        # Nie zaczynamy od 0, żeby mieć sensowne "poprzednie" miasto (i-1)
        for i in range(1, n - 2):
            for k in range(i + 1, n - 1):

                # Rozważamy dwie aktualne krawędzie w trasie: (a->b) oraz (c->d)
                a, b = best[i - 1], best[i]
                c, d = best[k], best[(k + 1) % n]

                # Sprawdzamy, czy lepiej będzie usunąć (a->b) i (c->d)
                # i zamiast tego dodać (a->c) oraz (b->d).
                # Jeśli nowy układ jest krótszy, delta będzie ujemna.
                delta = (distances[a, c] + distances[b, d]) - (distances[a, b] + distances[c, d])

                # -1e-9 to mały próg bezpieczeństwa na błędy zmiennoprzecinkowe
                if delta < -1e-9:
                    # Żeby uzyskać taki efekt połączeń, odwracamy fragment trasy między b i c
                    # (czyli elementy od i do k włącznie).
                    best[i : k + 1] = reversed(best[i : k + 1])
                    improved = True
                    break

            if improved:
                break
                # Jeśli w całym przebiegu nie było żadnej poprawy, to znaczy że jesteśmy w lokalnym optimum 2-opt
        if not improved:
            break

    return best


def seeded_random_state(seed: int | None = None) -> random.Random:
    rng = random.Random()
    if seed is not None:
        rng.seed(seed)
    return rng
