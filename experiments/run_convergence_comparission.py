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

    RESULTS_DIR = os.path.join(os.path.dirname(__file__), "../results/convergence_comparison_2")
    os.makedirs(RESULTS_DIR, exist_ok=True)

    # Função para rodar N simulações e pegar curvas
    def run_simulations(alpha, mu, n_runs=15):
        curves = []
        for _ in range(n_runs):
            aoa = AOA(
                fitness_func=fitness_fn,
                dim=cfg.aoa.dim,
                lb=cfg.aoa.lb,
                ub=cfg.aoa.ub,
                pop_start=None,
                pop_size=cfg.aoa.pop_size,
                max_iter=cfg.aoa.max_iter,
                seed=None,
                alpha=alpha,
                mu=mu,
                mop_max=cfg.aoa.mop_max,
                mop_min=cfg.aoa.mop_min,
            )
            _, _, conv_curve = aoa.solve(verbose=False)
            curves.append(conv_curve)
        return curves

    # Rodar para parâmetros otimizados e originais
    curves_opt = run_simulations(cfg.aoa.alpha_opt, cfg.aoa.mu_opt)
    curves_orig = run_simulations(cfg.aoa.alpha_orig, cfg.aoa.mu_orig)
    curves_opt_time = run_simulations(cfg.aoa.alpha_opt_time, cfg.aoa.mu_opt_time)

    # Plotar
    plt.figure(figsize=(6, 8))

    # Curvas com alpha/mu otimizados (tons de laranja)
    for curve in curves_opt:
        plt.plot(curve, color="orange", alpha=0.3)

    # Curvas com alpha/mu originais (tons de azul)
    for curve in curves_orig:
        plt.plot(curve, color="blue", alpha=0.3)

    for curve in curves_opt_time:
        plt.plot(curve, color="green", alpha=0.3)

    # Médias de cada grupo para destacar
    mean_opt = np.mean(curves_opt, axis=0)
    mean_orig = np.mean(curves_orig, axis=0)
    mean_opt_time = np.mean(curves_opt_time, axis=0)
    plt.plot(mean_opt, color="darkorange", linewidth=2, label="Média (Otimizados)")
    plt.plot(mean_orig, color="darkblue", linewidth=2, label="Média (Originais)")
    plt.plot(mean_opt_time, color="darkgreen", linewidth=2, label="Média segunda Otimização")

    plt.xlabel("Iteração")
    plt.ylabel("Fitness")
    plt.title("Comparação de Curvas de Convergência")
    plt.grid(True)
    plt.yscale("log")  # escala log para diferenças grandes
    plt.legend()

    plt.savefig(os.path.join(RESULTS_DIR, "comparacao_convergencia.png"))
    plt.close()

    print(f"Gráfico de comparação salvo em {RESULTS_DIR}")

if __name__ == "__main__":
    main()
