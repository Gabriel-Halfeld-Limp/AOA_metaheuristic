import hydra
from omegaconf import DictConfig
from functions import sphere, fob
from aoa_metaheuristic.optimizer import AOA
import numpy as np
import pandas as pd
import os
import yaml
import matplotlib.pyplot as plt

FUNC_MAP = {
    "sphere": sphere,
    "fob": fob,
}
@hydra.main(version_base=None, config_path="../config", config_name="config_tuning_time")
def main(cfg: DictConfig):
    fitness_fn = FUNC_MAP[cfg.aoa.fitness_func]
    #pop_start = pd.read_csv(cfg.aoa.pop_start).to_numpy()
    aoa = AOA(
        fitness_func=fitness_fn,
        dim=cfg.aoa.dim,
        lb=cfg.aoa.lb,
        ub=cfg.aoa.ub,
        pop_start=None,
        pop_size=cfg.aoa.pop_size,
        max_iter=cfg.aoa.max_iter,
        seed=None,
        alpha=cfg.aoa.alpha,
        mu=cfg.aoa.mu,
        mop_max=cfg.aoa.mop_max,
        mop_min=cfg.aoa.mop_min,
    )
    best_fit_all = []
    best_sol_all = []
    conv_curve_all = []
    elapsed_time_all = []

    for i in range(1000):
        print(i)
        bfit, best_sol, cc, time = aoa.solve_with_time(verbose=False)
        best_fit_all.append(bfit)
        best_sol_all.append(best_sol)
        conv_curve_all.append(cc)
        elapsed_time_all.append(time)
    
    # Converter para arrays
    best_fit_arr = np.array(best_fit_all)
    elapsed_time_arr = np.array(elapsed_time_all)
    sol_arr = np.array(best_sol_all)

    # Estatísticas
    stats = {
        "mean_fitness": float(np.mean(best_fit_arr)),
        "std_fitness": float(np.std(best_fit_arr)),
        "median_fitness": float(np.median(best_fit_arr)),
        "q1_fitness": float(np.percentile(best_fit_arr, 25)),
        "q3_fitness": float(np.percentile(best_fit_arr, 75)),
        "min_fitness": float(np.min(best_fit_arr)),
        "max_fitness": float(np.max(best_fit_arr)),
        "mean_time": float(np.mean(elapsed_time_arr)),
        "std_time": float(np.std(elapsed_time_arr)),
        "total_runs": len(best_fit_arr),
        "Mean_sol": np.mean(sol_arr, axis=0).tolist()
    }

    # Pasta para salvar resultados
    RESULTS_DIR = os.path.join(os.path.dirname(__file__), "../results/boxplot_optimized_tuning_time_escala_normal")
    os.makedirs(RESULTS_DIR, exist_ok=True)

    # Salvar estatísticas em YAML
    with open(os.path.join(RESULTS_DIR, "summary.yaml"), "w") as f:
        yaml.dump(stats, f)

    plt.figure(figsize=(6,8))
    box = plt.boxplot(
        best_fit_arr,
        vert=True,
        showmeans=True,
        patch_artist=True,
        flierprops=dict(marker='o', color='red', markersize=5)
    )

    # Legenda manual
    plt.plot([], [], color='orange', label='Mediana')
    plt.plot([], [], color='green', marker='^', linestyle='None', markersize=8, label='Média')
    plt.plot([], [], color='black', marker='o', linestyle='None', markersize=5, label='Outliers')
    plt.legend()

    plt.title("Boxplot do Fitness - 1000 simulações")
    plt.ylabel("Fitness")
    plt.grid(True)
    ax = plt.gca()
    #ax.set_yscale('log')
    ax.ticklabel_format(axis='y', style='plain')
    plt.savefig(os.path.join(RESULTS_DIR, "boxplot_fitness.png"))
    plt.close()

    print("Boxplot salvo e estatísticas armazenadas em summary.yaml")

if __name__ == "__main__":
    main()