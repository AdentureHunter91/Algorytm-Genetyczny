from __future__ import annotations

from typing import List, Tuple

import matplotlib.pyplot as plt


def plot_route(coords: List[Tuple[float, float]], route: List[int]):
    fig, ax = plt.subplots()
    xs = [coords[i][0] for i in route] + [coords[route[0]][0]]
    ys = [coords[i][1] for i in route] + [coords[route[0]][1]]
    ax.plot(xs, ys, marker="o")
    ax.set_title("Najlepsza trasa")
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    return fig


def plot_convergence(history: List[float]):
    fig, ax = plt.subplots()
    ax.plot(history, color="green")
    ax.set_title("Zbieżność algorytmu")
    ax.set_xlabel("Generacja")
    ax.set_ylabel("Długość trasy")
    return fig
