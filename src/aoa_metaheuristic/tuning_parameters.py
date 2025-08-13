import optuna
import hydra
from omegaconf import DictConfig
from functions import sphere, fob
from aoa_metaheuristic.optimizer import AOA
import numpy as np
import random

FUNC_MAP = {
    "sphere": sphere,
    "fob": fob,
}

@hydra.main(version_base=None, config_path="../../config", config_name="config")
def main(cfg: DictConfig):
    def objective(trial):
        alpha = trial.suggest_float("alpha", 1.0, 10.0)
        mu = trial.suggest_float("mu", 0.1, 1.0)
        fitness_fn = FUNC_MAP[cfg.aoa.fitness_func]
        seeds = random.sample(range(1_000_000), 10)  # 5 seeds aleatórias diferentes a cada trial
        results = []
        for seed in seeds:
            aoa = AOA(
                fitness_func=fitness_fn,
                dim=cfg.aoa.dim,
                lb=cfg.aoa.lb,
                ub=cfg.aoa.ub,
                pop_start=cfg.aoa.pop_start,
                pop_size=cfg.aoa.pop_size,
                max_iter=cfg.aoa.max_iter,
                seed=seed,
                alpha=alpha,
                mu=mu,
                mop_max=cfg.aoa.mop_max,
                mop_min=cfg.aoa.mop_min,
            )
            best_fit, _, _ = aoa.solve(verbose=False)
            results.append(best_fit)
        return float(np.mean(results))

    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=1000)

    print("Melhores hiperparâmetros encontrados:", study.best_params)
    print("Melhor fitness:", study.best_value)

if __name__ == "__main__":
    main()