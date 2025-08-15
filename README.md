# AOA Metaheuristic

This project implements and evaluates the **Arithmetic Optimization Algorithm (AOA)** for optimization problems, with code in both Python and MATLAB.
The original paper is from ABUALIGAH, Laith. ABUALIGAH, Laith et al. The arithmetic optimization algorithm. Computer methods in applied mechanics and engineering, v. 376, p. 113609, 2021.

## Project Structure

```
src/
  main.py                # Main Python script
  aoa_metaheuristic/     # AOA implementation in Python
  functions/             # Objective functions (e.g., sphere, fob)
  AOA/                   # MATLAB implementation of AOA

data/
  pop_start.csv          # Initial population (optional)

config/
  config.yaml            # Default Hydra configuration
  config_tuning_time.yaml# Configuration for time-penalized tuning

experiments/
  run_boxplot.py         # Script to generate boxplot of results
  run_convergence.py     # Script for convergence curve
  run_tuning.py          # Hyperparameter tuning script (Optuna)
  run_one_off.py         # Single AOA run
  ...
  
results/
  ...                    # Experiment results (images, CSV, YAML)
```

## Installation

1. **Clone the repository:**
   ```sh
   git clone <repo-url>
   cd AOA_metaheuristic
   ```

2. **Install dependencies with Poetry:**
   ```sh
   poetry install
   ```

3. **Activate the Poetry environment:**
   ```sh
   poetry shell
   ```

## How to Run

### 1. **Default execution**
```sh
python src/main.py
```

### 2. **Experiments**
- **Boxplot of results:**
  ```sh
  python experiments/run_boxplot.py
  ```
- **Convergence curve:**
  ```sh
  python experiments/run_convergence.py
  ```
- **Hyperparameter tuning (Optuna):**
  ```sh
  python experiments/run_tuning.py
  ```

### 3. **Configuration**
- Experiment parameters are defined in YAML files in `config/`.
- You can adjust population, bounds, objective function, hyperparameters, etc.

## Main Dependencies

- [Poetry](https://python-poetry.org/)
- [Hydra](https://hydra.cc/) (configuration management)
- [Optuna](https://optuna.org/) (hyperparameter tuning)
- [NumPy](https://numpy.org/)
- [Pandas](https://pandas.pydata.org/)
- [Matplotlib](https://matplotlib.org/)

## Results

- Experiment results are saved in `results/` (images, curves, statistics in YAML/CSV).
- Analysis and visualization scripts are in `experiments/`.
