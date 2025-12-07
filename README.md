# Algorytm genetyczny dla TSP

Interaktywna aplikacja Streamlit rozwiązująca problem komiwojażera za pomocą algorytmu genetycznego.

## Wymagania

Python 3.10+ oraz zależności z pliku `requirements.txt`:

```bash
pip install -r requirements.txt
```

## Struktura
- `app.py` – aplikacja Streamlit do uruchomienia z GUI.
- `tsp_problem.py` – ładowanie plików TSPLIB i tworzenie macierzy odległości.
- `genetic_algorithm.py` – implementacja algorytmu genetycznego.
- `operators.py` – operatory selekcji, krzyżowania i mutacji.
- `visualization.py` – wykresy trasy i zbieżności.
- `data/` – umieść pliki `att48.tsp` i/lub `berlin52.tsp` z TSPLIB.

## Uruchomienie

Pobierz wymagane pliki z TSPLIB do katalogu `data/`, a następnie:

```bash
streamlit run app.py
```

W panelu bocznym wybierz parametry algorytmu, a następnie kliknij **START**. Aplikacja wyświetli najlepszą dotychczasową trasę oraz wykres zbieżności w czasie rzeczywistym.
