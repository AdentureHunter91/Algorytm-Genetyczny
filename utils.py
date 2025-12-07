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
    total = 0.0
    for i in range(len(route)):
        total += distances[route[i - 1], route[i]]
    return total


def seeded_random_state(seed: int | None = None) -> random.Random:
    rng = random.Random()
    if seed is not None:
        rng.seed(seed)
    return rng
