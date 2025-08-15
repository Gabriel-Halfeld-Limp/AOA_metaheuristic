import os
import numpy as np
from aoa_metaheuristic.optimizer import AOA
from functions import fob

from functions import sphere, fob
from aoa_metaheuristic.optimizer import AOA
import hydra
from omegaconf import DictConfig
import pandas as pd

FUNC_MAP = {
    "sphere": sphere,
    "fob": fob,
}

#Cria pasta results se nao existe e salva dentro de convergence
RESULTS_DIR = os.path.join(os.path.dirname(__file__), "../results/convergence_curve")
os.makedirs(RESULTS_DIR, exist_ok=True)

@hydra.main(version_base=None, config_path="../config", config_name="config")
def main(cfg: DictConfig):
    # Obter função real do mapa
    fitness_fn = FUNC_MAP.get(cfg.aoa.fitness_func)
    if fitness_fn is None:
        raise ValueError(f"Função de fitness '{cfg.aoa.fitness_func}' não encontrada em FUNC_MAP")

    pop_start = pd.read_csv(cfg.aoa.pop_start).to_numpy()
    aoa = AOA(
        fitness_func=fitness_fn,
        dim=cfg.aoa.dim,
        lb=cfg.aoa.lb,
        ub=cfg.aoa.ub,
        pop_start=pop_start,
        pop_size=cfg.aoa.pop_size,
        max_iter=cfg.aoa.max_iter,
        seed=cfg.aoa.seed,
        alpha=cfg.aoa.alpha,
        mu=cfg.aoa.mu,
        mop_max=cfg.aoa.mop_max,
        mop_min=cfg.aoa.mop_min,
    )

    best_sol, best_fit, conv_curve = aoa.solve(verbose=False)

    conv_file = os.path.join(RESULTS_DIR, "conv_curve.csv")
    np.savetxt(conv_file, conv_curve, delimiter=",")
    print(f"Curva de convergência salva em: {conv_file}")

    # Salva melhor fitness e solução
    summary_file = os.path.join(RESULTS_DIR, "summary.txt")
    with open(summary_file, "w") as f:
        f.write(f"Melhor fitness: {best_fit}\n")
        f.write(f"Melhor solução: {best_sol}\n")
    print(f"Resumo salvo em: {summary_file}")

    # Salva o gráfico usando o método interno da classe
    plot_file = os.path.join(RESULTS_DIR, "conv_curve.png")
    aoa.plot_convergence(save_path=plot_file)

if __name__ == "__main__":
    main()