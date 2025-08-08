import numpy as np


import numpy as np

class AOA:
    def __init__(self, fitness_func, bounds, dim, pop_size=30, max_iter=1000, alpha=5, mu=0.5,
                 mop_max=1, mop_min=0.2, seed=10):

        # Um único gerador com seed mestre
        self.rng = np.random.default_rng(seed)

        self.fitness_func = fitness_func
        self.bounds = np.array(bounds)
        self.dim = dim
        self.pop_size = pop_size
        self.max_iter = max_iter
        self.alpha = alpha
        self.mu = mu

        self.lb = self.bounds[:, 0]
        self.ub = self.bounds[:, 1]

        self.population = self.rng.uniform(self.lb, self.ub, (self.pop_size, self.dim))
        self.fitness = np.array(list(map(self.fitness_func, self.population)))

        self.best_idx = np.argmin(self.fitness)
        self.best_solution = self.population[self.best_idx].copy()
        self.best_fitness = self.fitness[self.best_idx]

        self.mop_max = mop_max
        self.mop_min = mop_min
        self.eps = 1e-8
        self.conv_curve = []

    def _moa(self, iteration):
        # MOA varia linearmente entre mop_min e mop_max durante as iterações
        return self.mop_min + iteration * (self.mop_max - self.mop_min) / self.max_iter

    def _mop(self, iteration):
        # MOP decresce seguindo potência (alpha) durante as iterações
        return 1 - (iteration ** (1 / self.alpha)) / (self.max_iter ** (1 / self.alpha))

    def _division(self, best, mop, diff):
        return best / (mop + self.eps) * diff

    def _multiplication(self, best, mop, diff):
        return best * mop * diff

    def _sum(self, best, mop, diff):
        return best - mop * diff

    def _subtraction(self, best, mop, diff):
        return best + mop * diff

    def _update_position(self, best, j, moa, mop):
        diff = (self.ub[j] - self.lb[j]) * self.mu + self.lb[j]
        r1 = self.rng.random()
        if r1 < moa:
            r2 = self.rng.random()
            if r2 > 0.5:
                new_val = self._division(best[j], mop, diff)
            else:
                new_val = self._multiplication(best[j], mop, diff)
        else:
            r3 = self.rng.random()
            if r3 > 0.5:
                new_val = self._sum(best[j], mop, diff)
            else:
                new_val = self._subtraction(best[j], mop, diff)

        # Garante que fica dentro dos limites
        return np.clip(new_val, self.lb[j], self.ub[j])

    def optimize(self, verbose=False):
        for iteration in range(1, self.max_iter + 1):
            moa = self._moa(iteration)
            mop = self._mop(iteration)

            Xnew = np.copy(self.population)

            for i in range(self.pop_size):
                for j in range(self.dim):
                    Xnew[i, j] = self._update_position(self.best_solution, j, moa, mop)

                new_fitness = self.fitness_func(Xnew[i])
                if new_fitness < self.fitness[i]:
                    self.population[i] = Xnew[i]
                    self.fitness[i] = new_fitness

                if self.fitness[i] < self.best_fitness:
                    self.best_fitness = self.fitness[i]
                    self.best_solution = self.population[i].copy()

            self.conv_curve.append(self.best_fitness)

            if verbose and iteration % 100 == 0:
                print(f"Iter {iteration}: Best fitness = {self.best_fitness:.6f}")

        return self.best_solution, self.best_fitness, self.conv_curve
