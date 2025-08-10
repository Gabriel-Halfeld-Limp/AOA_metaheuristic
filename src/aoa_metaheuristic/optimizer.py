import numpy as np
import matplotlib.pyplot as plt
from typing import Callable, Optional, Tuple, List

class AOA:
    def __init__(
        self,
        fitness_func: Callable,
        dim: int,
        ub,
        lb,
        pop_start: Optional[np.ndarray] = None,
        pop_size: int = 30,
        max_iter: int = 100,
        seed: int = 10,
        alpha: float = 5,
        mu: float = 0.5,
        mop_max: float = 1,
        mop_min: float = 0.2,
    ):

        # RNG
        self.rng = np.random.default_rng(seed)

        # Problem
        self.fitness_func = fitness_func
        self.dim = int(dim)
        self.lb = np.array(lb, dtype=float)
        self.ub = np.array(ub, dtype=float)

        # Optimizer params
        self.pop_size = int(pop_size)
        self.max_iter = int(max_iter)
        self.alpha = float(alpha)
        self.mu = float(mu)
        self.mop_max = float(mop_max)
        self.mop_min = float(mop_min)

        # constants
        self.eps = 1e-12
        self.moa_constant = (self.mop_max - self.mop_min) / self.max_iter
        self.mop_constant = self.max_iter ** (1 / self.alpha)

        # --- Expand scalar bounds to vectors BEFORE computing diffs ---
        if self.ub.size == 1:
            self.ub = np.full(self.dim, float(self.ub.item()))
        if self.lb.size == 1:
            self.lb = np.full(self.dim, float(self.lb.item()))

        # precompute diffs per dimension (constant)
        self.diffs = (self.ub - self.lb) * self.mu + self.lb

        # initialize population (pop_start validation)
        if pop_start is not None:
            self.population = np.array(pop_start, dtype=float)
            if self.population.ndim != 2 or self.population.shape[1] != self.dim:
                raise ValueError("pop_start must be shape (pop_size, dim) or None.")
            self.pop_size = self.population.shape[0]
        else:
            self.population = self.rng.random((self.pop_size, self.dim)) * (self.ub - self.lb) + self.lb

        # fitness (use _eval so fobs that accept only lists still work)
        self.fitness = np.array([self._eval(self.population[i]) for i in range(self.pop_size)], dtype=float)

        # best
        self.best_idx = int(np.argmin(self.fitness))
        self.best_solution = self.population[self.best_idx].copy()
        self.best_fitness = float(self.fitness[self.best_idx])

        # convergence
        self.conv_curve: List[float] = []

    def _eval(self, x: np.ndarray) -> float:
        try:
            return float(self.fitness_func(x))
        except Exception:
            return float(self.fitness_func(x.tolist()))

    def _mop(self, iteration: int) -> float:
        return 1.0 - (iteration ** (1.0 / self.alpha)) / self.mop_constant

    def _moa(self, iteration: int) -> float:
        return self.mop_min + iteration * self.moa_constant

    # primitive operators (per-coordinate)
    def _division(self, best_j: float, mop: float, diff_j: float) -> float:
        return best_j / (mop + self.eps) * diff_j

    def _multiplication(self, best_j: float, mop: float, diff_j: float) -> float:
        return best_j * mop * diff_j

    def _subtraction(self, best_j: float, mop: float, diff_j: float) -> float:
        return best_j - mop * diff_j

    def _addition(self, best_j: float, mop: float, diff_j: float) -> float:
        return best_j + mop * diff_j

    def _update_position(self, i: int, j: int, mop: float, moa: float) -> float:
        r1 = self.rng.random()
        best_j = self.best_solution[j]
        #diff_j = self.diffs[j]
        diff_j = (self.ub[j] - self.lb[j]) * self.mu

        if r1 < moa:
            if self.rng.random() > 0.5:
                return self._division(best_j, mop, diff_j)
            else:
                return self._multiplication(best_j, mop, diff_j)
        else:
            if self.rng.random() > 0.5:
                return self._subtraction(best_j, mop, diff_j)
            else:
                return self._addition(best_j, mop, diff_j)

    def __evaluate_population(self, new_population: np.ndarray) -> None:
        for i in range(self.pop_size):
            new_population[i] = np.clip(new_population[i], self.lb, self.ub)
            new_f = self._eval(new_population[i])
            if new_f < self.fitness[i]:
                self.population[i] = new_population[i]
                self.fitness[i] = new_f
                if new_f < self.best_fitness:
                    self.best_fitness = new_f
                    self.best_solution = new_population[i].copy()
    def _evaluate_population(self, new_population: np.ndarray) -> None:
        for i in range(self.pop_size):
            new_population[i] = np.clip(new_population[i], self.lb, self.ub)
            new_f = self._eval(new_population[i])
            if new_f < self.fitness[i]:
                self.population[i] = new_population[i]
                self.fitness[i] = new_f
            # Atualize o best_solution SEMPRE que encontrar um fitness melhor, mesmo que não tenha melhorado o indivíduo
            if self.fitness[i] < self.best_fitness:
                self.best_fitness = self.fitness[i]
                self.best_solution = self.population[i].copy()

    def solve(self, verbose: bool = False, print_every: int = 10) -> Tuple[np.ndarray, float, List[float]]:
        for iteration in range(1, self.max_iter + 1):
            mop = self._mop(iteration)
            moa = self._moa(iteration)

            new_population = np.copy(self.population)
            for i in range(self.pop_size):
                for j in range(self.dim):
                    new_population[i, j] = self._update_position(i, j, mop, moa)

            self._evaluate_population(new_population)
            self.conv_curve.append(self.best_fitness)

            if verbose and (iteration % print_every == 0 or iteration == 1 or iteration == self.max_iter):
                print(f"Iteration {iteration}, Best fitness: {self.best_fitness:.6e}")

        return self.best_solution, self.best_fitness, self.conv_curve

    def plot_convergence(self, ax=None, figsize=(8,4)):
        if ax is None:
            fig, ax = plt.subplots(figsize=figsize)
        else:
            fig = None
        ax.plot(self.conv_curve, label="best")
        ax.set_xlabel("Iteration")
        ax.set_ylabel("Best fitness")
        ax.grid(True)
        ax.legend()
        if fig is not None:
            plt.show()
