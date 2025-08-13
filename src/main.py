from functions import sphere, fob
from aoa_metaheuristic.optimizer import AOA
import hydra
from omegaconf import DictConfig

FUNC_MAP = {
    "sphere": sphere,
    "fob": fob,
}

@hydra.main(version_base=None, config_path="../config", config_name="config")
def main(cfg: DictConfig):
    # Obter função real do mapa
    fitness_fn = FUNC_MAP.get(cfg.aoa.fitness_func)
    if fitness_fn is None:
        raise ValueError(f"Função de fitness '{cfg.aoa.fitness_func}' não encontrada em FUNC_MAP")

    aoa = AOA(
        fitness_func=fitness_fn,
        dim=cfg.aoa.dim,
        lb=cfg.aoa.lb,
        ub=cfg.aoa.ub,
        pop_start=cfg.aoa.pop_start,
        pop_size=cfg.aoa.pop_size,
        max_iter=cfg.aoa.max_iter,
        seed=cfg.aoa.seed,
        alpha=cfg.aoa.alpha,
        mu=cfg.aoa.mu,
        mop_max=cfg.aoa.mop_max,
        mop_min=cfg.aoa.mop_min,
    )

    best_sol, best_fit, _ = aoa.solve(verbose=cfg.aoa.verbose)
    print("Melhor solução:", best_sol)
    print("Melhor fitness:", best_fit)
    #aoa.plot_convergence()

if __name__ == "__main__":
    main()