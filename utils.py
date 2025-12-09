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
    if len(route) == 0:
        return float("inf")
    arr = np.asarray(route, dtype=int)
    return float(np.sum(distances[arr, np.roll(arr, -1)]))


def two_opt(route: List[int], distances: np.ndarray, max_iterations: int = 10) -> List[int]:
    """Prosta lokalna optymalizacja 2-opt z wczesnym zatrzymaniem."""

    if len(route) < 4:
        return route

    best = route.copy()
    n = len(best)

    for _ in range(max_iterations):
        improved = False
        for i in range(1, n - 2):
            for k in range(i + 1, n - 1):
                a, b = best[i - 1], best[i]
                c, d = best[k], best[(k + 1) % n]

                delta = (distances[a, c] + distances[b, d]) - (distances[a, b] + distances[c, d])
                if delta < -1e-9:
                    best[i : k + 1] = reversed(best[i : k + 1])
                    improved = True
                    break
            if improved:
                break
        if not improved:
            break

    return best


def seeded_random_state(seed: int | None = None) -> random.Random:
    rng = random.Random()
    if seed is not None:
        rng.seed(seed)
    return rng
