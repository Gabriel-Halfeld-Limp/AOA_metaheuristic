import numpy as np
from typing import Callable, Optional

class AOA_3:
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
        mu: float = 0.499,
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
        self.eps = np.finfo(float).eps
    
    def _eval(self, x: np.ndarray) -> float:
        try:
            return float(self.fitness_func(x))
        except Exception:
            return float(self.fitness_func(x.tolist()))

    def initialization(self):
        """
        Inicializa a população de forma uniforme entre os limites lb e ub.
        ub e lb podem ser escalares ou arrays de tamanho dim.
        """
        if self.ub.size == 1:  # limites iguais para todas as variáveis
            pop = self.rng.random((self.pop_size, self.dim)) * (self.ub - self.lb) + self.lb
        else:  # limites diferentes por variável
            pop = np.zeros((self.pop_size, self.dim))
            for i in range(self.dim):
                pop[:, i] = self.rng.random(self.pop_size) * (self.ub[i] - self.lb[i]) + self.lb[i]
        return pop

    def solve(self, verbose=True):
        print("AOA Working")

        pop_best = np.zeros(self.dim)
        fitness_best = np.inf
        conv_curve = np.zeros(self.max_iter)

        # Inicializa posições
        pop = self.initialization()
        #print(f"Primeira população:{pop}")
        pop_new = np.copy(pop)

        fitness = np.zeros(self.pop_size)
        fitness_new = np.zeros(self.pop_size)

        # Avaliação inicial
        for i in range(self.pop_size):
            fitness[i] = self._eval(pop[i, :])
            #print(f"Primeiro fitness:{fitness}")
            if fitness[i] < fitness_best:
                fitness_best = fitness[i]
                pop_best = np.copy(pop[i, :])

        for iter in range(1, self.max_iter + 1):
            mop = 1 - ((iter) ** (1 / self.alpha) / (self.max_iter) ** (1 / self.alpha))
            moa = self.mop_min + iter * ((self.mop_max - self.mop_min) / self.max_iter)

            for i in range(self.pop_size):
                for j in range(self.dim):
                    r1 = self.rng.random()
                    #print(f"r1:{r1}")

                    if np.size(self.lb) == 1:  # limites iguais
                        if r1 < moa:
                            r2 = self.rng.random()
                            #print(f"r2:{r2}")
                            if r2 > 0.5:
                                pop_new[i, j] = pop_best[j] / (mop + self.eps) * ((self.ub - self.lb) * self.mu + self.lb)
                            else:
                                pop_new[i, j] = pop_best[j] * mop * ((self.ub - self.lb) * self.mu + self.lb)
                        else:
                            r3 = self.rng.random()
                            #print(f"r3:{r3}")
                            if r3 > 0.5:
                                pop_new[i, j] = pop_best[j] - mop * ((self.ub - self.lb) * self.mu + self.lb)
                            else:
                                pop_new[i, j] = pop_best[j] + mop * ((self.ub - self.lb) * self.mu + self.lb)
                    else:  # limites diferentes por variável
                        if r1 < moa:
                            r2 = self.rng.random()
                            if r2 > 0.5:
                                pop_new[i, j] = pop_best[j] / (mop + self.eps) * ((self.ub[j] - self.lb[j]) * self.mu + self.lb[j])
                            else:
                                pop_new[i, j] = pop_best[j] * mop * ((self.ub[j] - self.lb[j]) * self.mu + self.lb[j])
                        else:
                            r3 = self.rng.random()
                            if r3 > 0.5:
                                pop_new[i, j] = pop_best[j] - mop * ((self.ub[j] - self.lb[j]) * self.mu + self.lb[j])
                            else:
                                pop_new[i, j] = pop_best[j] + mop * ((self.ub[j] - self.lb[j]) * self.mu + self.lb[j])

                # Correção de limites
                if np.size(self.lb) == 1:
                    pop_new[i, :] = np.clip(pop_new[i, :], self.lb, self.ub)
                else:
                    pop_new[i, :] = np.minimum(np.maximum(pop_new[i, :], self.lb), self.ub)

                # Avaliação
                fitness_new[i] = self._eval(pop_new[i, :])
                if fitness_new[i] < fitness[i]:
                    pop[i, :] = pop_new[i, :]
                    fitness[i] = fitness_new[i]
                if fitness[i] < fitness_best:
                    fitness_best = fitness[i]
                    pop_best = np.copy(pop[i, :])
            
                #print(f"Population:{pop}")
                #print(f"Fitness:{fitness}")

            conv_curve[iter - 1] = fitness_best
            if verbose:
                if iter % 5 == 0:
                    print(f"At iteration {iter} the best solution fitness is {fitness_best}")

        return fitness_best, pop_best, conv_curve