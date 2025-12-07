from __future__ import annotations

from pathlib import Path
from typing import List, Tuple

import numpy as np
import tsplib95

from utils import distance_matrix


class TSP:
    def __init__(self, filepath: str):
        self.filepath = Path(filepath)
        self.name: str | None = None
        self.dimension: int = 0
        self.edge_weight_type: str = "EUC_2D"
        self.coordinates: List[Tuple[float, float]] = []
        self.distances: np.ndarray | None = None
        self._load()

    def _load(self) -> None:
        if not self.filepath.exists():
            raise FileNotFoundError(f"Dataset file not found: {self.filepath}")
        problem = tsplib95.load(str(self.filepath))
        self.name = problem.name
        self.dimension = problem.dimension
        self.edge_weight_type = problem.edge_weight_type
        self.coordinates = [problem.node_coords[i + 1] for i in range(self.dimension)]
        self.distances = distance_matrix(self.coordinates, self.edge_weight_type)

    def route_length(self, route: List[int]) -> float:
        if self.distances is None:
            raise ValueError("Distance matrix has not been initialized")
        return float(sum(self.distances[route[i - 1], route[i]] for i in range(len(route))))
