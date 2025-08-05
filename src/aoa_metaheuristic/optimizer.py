import numpy as np

class AOA:
    def __init__(self, fitness_func, bounds, dim, pop_size=30, max_iter=1000, alpha=5, mu=0.5):
        self.fitness_func = fitness_func
        self.bounds = np.array(bounds)
        self.dim = dim
        self.pop_size = pop_size
        self.max_iter = max_iter
        self.alpha = alpha
        self.mu = mu

        self.lb = self.bounds[:, 0]
        self.ub = self.bounds[:, 1]
        self.population = np.random.uniform(self.lb, self.ub, (self.pop_size, self.dim))
        self.fitness = np.apply_along_axis(self.fitness_func, 1, self.population)
        self.best_idx = np.argmin(self.fitness)
        self.best_solution = self.population[self.best_idx].copy()
        self.best_fitness = self.fitness[self.best_idx]

    def _moa(self, iteration):
        return 0.2 + iteration * (0.9 - 0.2) / self.max_iter

    def _mop(self, iteration):
        return 1 - (iteration ** (1 / self.alpha)) / (self.max_iter ** (1 / self.alpha))

    def _update_position(self, current, best, j, moa, mop):
        r1, r2, r3 = np.random.rand(3)
        diff = (self.ub[j] - self.lb[j]) * self.mu + self.lb[j]

        if r1 > moa:
            if r2 < 0.5:
                new_val = best[j] / (mop + 1e-8) * diff
            else:
                new_val = best[j] * mop * diff
        else:
            if r3 < 0.5:
                new_val = best[j] - mop * diff
            else:
                new_val = best[j] + mop * diff

        return np.clip(new_val, self.lb[j], self.ub[j])

    def optimize(self, verbose=False):
        for iteration in range(1, self.max_iter + 1):
            moa = self._moa(iteration)
            mop = self._mop(iteration)

            for i in range(self.pop_size):
                for j in range(self.dim):
                    self.population[i, j] = self._update_position(
                        self.population[i], self.best_solution, j, moa, mop
                    )

            self.fitness = np.apply_along_axis(self.fitness_func, 1, self.population)
            current_best_idx = np.argmin(self.fitness)
            if self.fitness[current_best_idx] < self.best_fitness:
                self.best_fitness = self.fitness[current_best_idx]
                self.best_solution = self.population[current_best_idx].copy()

            if verbose and iteration % 100 == 0:
                print(f"Iter {iteration}: Best fitness = {self.best_fitness:.6f}")

        return self.best_solution, self.best_fitness
