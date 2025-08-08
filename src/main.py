from fob_func import fob
from aoa_metaheuristic.optimizer import AOA
import hydra
from omegaconf import DictConfig

def fitness_wrapper(x):
    return fob(x.tolist())  # converte np.ndarray para list

@hydra.main(config_path="../config", config_name="config")
def main(cfg: DictConfig):
    # usar fitness_wrapper no AOA
    aoa = AOA(
        fitness_func=fitness_wrapper,
        bounds=cfg.aoa.bounds,
        dim=cfg.aoa.dim,
        pop_size=cfg.aoa.pop_size,
        max_iter=cfg.aoa.max_iter,
        alpha=cfg.aoa.alpha,
        mu=cfg.aoa.mu,
        mop_max=cfg.aoa.mop_max,
        mop_min=cfg.aoa.mop_min,
        seed=cfg.aoa.seed,
    )

    best_sol, best_fit, conv_curve = aoa.optimize(verbose=True)
    print("Melhor solução:", best_sol)
    print("Melhor fitness:", best_fit)

if __name__ == "__main__":
    main()