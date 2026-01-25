from __future__ import annotations

import time
from pathlib import Path

import streamlit as st

from genetic_algorithm import GeneticAlgorithm
from tsp_problem import TSP
from visualization import plot_convergence, plot_route

DATASETS = {
    "ATT48": "data/att48.tsp",
    "Berlin52": "data/berlin52.tsp",
}


st.set_page_config(page_title="Algorytm genetyczny - TSP", layout="wide")
st.title("Problem komiwojażera - Algorytm genetyczny")

st.sidebar.header("Parametry")
pop_size = st.sidebar.slider("Rozmiar populacji", min_value=50, max_value=500, value=150, step=10)
generations = st.sidebar.slider("Liczba generacji", min_value=100, max_value=2000, value=600, step=50)
mutation_prob = st.sidebar.slider("Prawdopodobieństwo mutacji", min_value=0.01, max_value=0.2, value=0.05, step=0.01)
crossover_prob = st.sidebar.slider("Prawdopodobieństwo krzyżowania", min_value=0.5, max_value=1.0, value=0.9, step=0.05)
dataset_name = st.sidebar.selectbox("Dataset", list(DATASETS.keys()))
selection_method = st.sidebar.selectbox("Metoda selekcji", ["tournament", "roulette"])
crossover_method = st.sidebar.selectbox("Metoda krzyżowania", ["ox", "pmx"])
mutation_method = st.sidebar.selectbox("Metoda mutacji", ["swap", "inversion"])
elite_ratio = st.sidebar.slider("Rozmiar elity (%)", min_value=0, max_value=20, value=5, step=1) / 100
two_opt_enabled = st.sidebar.checkbox("Optymalizacja 2-opt", value=True)
two_opt_prob = 0.0
if two_opt_enabled:
    two_opt_prob = st.sidebar.slider(
        "Prawdopodobieństwo 2-opt", min_value=0.0, max_value=1.0, value=0.2, step=0.05
    )
start = st.sidebar.button("START")

status_text = st.empty()
progress_bar = st.progress(0)
col1, col2 = st.columns(2)
route_placeholder = col1.empty()
convergence_placeholder = col2.empty()
metrics_placeholder = st.empty()


def load_problem(name: str) -> TSP | None:
    path = Path(DATASETS[name])
    if not path.exists():
        st.error(f"Brak pliku datasetu: {path}. Dodaj plik TSPLIB do katalogu data/")
        return None
    return TSP(str(path))


if start:
    problem = load_problem(dataset_name)
    if problem:
        ga = GeneticAlgorithm(
            problem,
            population_size=pop_size,
            generations=generations,
            mutation_prob=mutation_prob,
            crossover_prob=crossover_prob,
            selection_method=selection_method,
            crossover_method=crossover_method,
            mutation_method=mutation_method,
            elite_ratio=elite_ratio,
           # seed = int(time.time() * 1000),
            seed=42,
            two_opt_prob=two_opt_prob,
        )

        start_time = time.time()
        live_history: list[float] = []

        def on_generation(gen: int, best: float, route: list[int]):
            live_history.append(best)
            if gen % 10 == 0 or gen == generations - 1:
                fig_route = plot_route(problem.coordinates, route)
                route_placeholder.pyplot(fig_route)
                fig_conv = plot_convergence(live_history)
                convergence_placeholder.pyplot(fig_conv)
                status_text.text(f"Generacja {gen+1}/{generations} - najlepsza długość: {best:.2f}")
                progress_bar.progress((gen + 1) / generations)
                time.sleep(0.01)

        ga_result = ga.run(callback=on_generation)
        elapsed = time.time() - start_time

        fig_route = plot_route(problem.coordinates, ga_result.best_route)
        route_placeholder.pyplot(fig_route)
        fig_conv = plot_convergence(ga_result.history)
        convergence_placeholder.pyplot(fig_conv)

        optimum = 10628 if dataset_name == "ATT48" else 7542
        metrics_placeholder.markdown(
            f"""
            **Najlepsza długość:** {ga_result.best_distance:.2f}  \
            **Różnica od optimum:** {ga_result.best_distance - optimum:.2f}  \
            **Generacje:** {generations}  \
            **Czas:** {elapsed:.2f}s  \
            **Poprawa względem startu:** {((ga_result.history[0] - ga_result.best_distance) / ga_result.history[0]) * 100:.2f}%
            """
        )

        status_text.success("Zakończono obliczenia")
        progress_bar.progress(1.0)
else:
    st.info("Ustaw parametry i kliknij START, aby rozpocząć.")
