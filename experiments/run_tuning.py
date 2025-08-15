import optuna
import hydra
from omegaconf import DictConfig
from functions import sphere, fob
from aoa_metaheuristic.optimizer import AOA
import numpy as np
import pandas as pd
import os
import yaml

FUNC_MAP = {
    "sphere": sphere,
    "fob": fob,
}
@hydra.main(version_base=None, config_path="../config", config_name="config")
def main(cfg: DictConfig):
    fitness_fn = FUNC_MAP[cfg.aoa.fitness_func]
    pop_start = pd.read_csv(cfg.aoa.pop_start).to_numpy()

    def optuna_objective(trial):
        alpha = trial.suggest_float("alpha", 1.0, 10.0)
        mu = trial.suggest_float("mu", 0.1, 1.0)
        results = []
        for _ in range(30):
            aoa = AOA(
                fitness_func=fitness_fn,
                dim=cfg.aoa.dim,
                lb=cfg.aoa.lb,
                ub=cfg.aoa.ub,
                pop_start=pop_start,
                pop_size=cfg.aoa.pop_size,
                max_iter=cfg.aoa.max_iter,
                seed=None,
                alpha=alpha,
                mu=mu,
                mop_max=cfg.aoa.mop_max,
                mop_min=cfg.aoa.mop_min,
            )
            best_fit, _, _ = aoa.solve(verbose=False)
            results.append(best_fit)
        mean_fit = np.mean(results)
        std_fit = np.std(results)
        return float(mean_fit + std_fit)

    study = optuna.create_study(direction="minimize")
    study.optimize(optuna_objective, n_trials=1000)

    print("Melhores hiperparâmetros encontrados:", study.best_params)
    print("Melhor fitness:", study.best_value)

    # Pasta para salvar resultados
    RESULTS_DIR = os.path.join(os.path.dirname(__file__), "../results/tuning_aoa")
    os.makedirs(RESULTS_DIR, exist_ok=True)

    # Salva melhores parâmetros e melhor fitness em YAML
    best_params_file = os.path.join(RESULTS_DIR, "best_params.yaml")
    with open(best_params_file, "w") as f:
        yaml.dump({"best_params": study.best_params, "best_value": float(study.best_value)}, f)

    print("Melhores hiperparâmetros encontrados:", study.best_params)
    print("Melhor fitness:", study.best_value)

    # Roda AOA final com melhores parâmetros
    best_alpha = study.best_params["alpha"]
    best_mu = study.best_params["mu"]

    aoa_final = AOA(
        fitness_func=fitness_fn,
        dim=cfg.aoa.dim,
        lb=cfg.aoa.lb,
        ub=cfg.aoa.ub,
        pop_start=pop_start,
        pop_size=cfg.aoa.pop_size,
        max_iter=cfg.aoa.max_iter,
        seed=10,
        alpha=best_alpha,
        mu=best_mu,
        mop_max=cfg.aoa.mop_max,
        mop_min=cfg.aoa.mop_min,
    )

    best_fit, best_sol, conv_curve = aoa_final.solve(verbose=False)

    # Salva a curva de convergência
    aoa_final.plot_convergence(save_path=os.path.join(RESULTS_DIR, "final_conv_curve.png"))
    np.savetxt(os.path.join(RESULTS_DIR, "final_conv_curve.csv"), conv_curve, delimiter=",")

    # Salva resumo da execução
    summary_file = os.path.join(RESULTS_DIR, "summary.yaml")
    with open(summary_file, "w") as f:
        yaml.dump({"best_fit": float(best_fit), "best_solution": best_sol.tolist()}, f)

if __name__ == "__main__":
    main()