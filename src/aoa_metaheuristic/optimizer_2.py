import numpy as np
import matplotlib.pyplot as plt
from typing import Callable, Optional, Tuple, List

class AOA_2:
    """AOA close to the original procedural implementation, translated to a class.

    Includes optional debugging instrumentation to help diagnose collapse-to-bound issues.
    """

    def __init__(
        self,
        fitness_func: Callable,
        dim: int,
        ub=1.0,
        lb=0.0,
        pop_start: Optional[np.ndarray] = None,
        pop_size: int = 30,
        max_iter: int = 100,
        seed: int = 10,
        alpha: float = 5.0,
        mu: float = 0.499,
        mop_max: float = 1.0,
        mop_min: float = 0.2,
    ):
        # RNG
        self.rng = np.random.default_rng(seed)

        # Problem
        self.fitness_func = fitness_func
        self.dim = int(dim)
        # store LB/UB exactly as provided (can be scalars or arrays) to match
        # the original code branching on np.size(LB) inside the loop
        self.LB = np.array(lb, dtype=float)
        self.UB = np.array(ub, dtype=float)

        # Optimizer params
        self.pop_size = int(pop_size)
        self.M_Iter = int(max_iter)
        self.Alpha = float(alpha)
        self.Mu = float(mu)
        self.MOP_Max = float(mop_max)
        self.MOP_Min = float(mop_min)

        # constants
        self.eps = np.finfo(float).eps

        # initialize population (pop_start validation)
        if pop_start is not None:
            self.X = np.array(pop_start, dtype=float)
            if self.X.ndim != 2 or self.X.shape[1] != self.dim:
                raise ValueError("pop_start must be shape (pop_size, dim) or None.")
            self.pop_size = self.X.shape[0]
        else:
            # use the same initialization rule from your reference
            if self.UB.size == 1:
                self.X = self.rng.random((self.pop_size, self.dim)) * (self.UB - self.LB) + self.LB
            else:
                self.X = np.zeros((self.pop_size, self.dim))
                for i in range(self.dim):
                    self.X[:, i] = self.rng.random(self.pop_size) * (self.UB[i] - self.LB[i]) + self.LB[i]

        # fitness evaluation arrays
        self.Ffun = np.zeros(self.pop_size)
        self.Ffun_new = np.zeros(self.pop_size)

        # evaluate initial population
        for i in range(self.pop_size):
            self.Ffun[i] = self._eval(self.X[i, :])

        # best
        self.Best_P = np.zeros(self.dim)
        self.Best_FF = np.inf
        for i in range(self.pop_size):
            if self.Ffun[i] < self.Best_FF:
                self.Best_FF = self.Ffun[i]
                self.Best_P = np.copy(self.X[i, :])

        # convergence
        self.Conv_curve = np.zeros(self.M_Iter)

    def _eval(self, x):
        """Permite que fitness_func aceite tanto listas quanto arrays."""
        try:
            return self.fitness_func(x)
        except Exception:
            return self.fitness_func(x.tolist())

    def _debug_state(self, it: int, X: np.ndarray, Xnew: np.ndarray, improvements: int) -> None:
        """Print debug information summarizing the population state."""
        # counts of entries equal to bounds (element-wise)
        count_at_ub = int(np.isclose(Xnew, self.UB).sum()) if self.UB.size > 1 else int(np.isclose(Xnew, self.UB).sum())
        count_at_lb = int(np.isclose(Xnew, self.LB).sum()) if self.LB.size > 1 else int(np.isclose(Xnew, self.LB).sum())
        # individuals fully at ub/lb
        full_at_ub = int(np.all(np.isclose(Xnew, self.UB), axis=1).sum())
        full_at_lb = int(np.all(np.isclose(Xnew, self.LB), axis=1).sum())

        print(f"DEBUG iter {it}: Best_FF={self.Best_FF:.6g}, improvements_this_iter={improvements}")
        print(f"  entries@UB={count_at_ub}, entries@LB={count_at_lb}, individuals@UB={full_at_ub}, individuals@LB={full_at_lb}")
        # show sample individuals
        sample_n = min(3, self.pop_size)
        for s in range(sample_n):
            print(f"  X[{s}] (old) min/max: {X[s].min():.6g}/{X[s].max():.6g}, Xnew[{s}] min/max: {Xnew[s].min():.6g}/{Xnew[s].max():.6g}")
        print(f"  Best_P sample: {np.round(self.Best_P[:min(6, self.dim)],4)}")

    def solve(self, verbose: bool = True, debug: bool = False, debug_every: int = 1) -> Tuple[float, np.ndarray, np.ndarray]:
        """Execute optimization preserving the original loop structure and returns
        (Best_FF, Best_P, Conv_curve) to match your functional code.

        Parameters
        ----------
        verbose: print regular progress every 5 iters
        debug: when True prints debug info every `debug_every` iterations
        debug_every: frequency of debug prints
        """
        X = np.copy(self.X)
        Xnew = np.copy(X)

        C_Iter = 1
        while C_Iter <= self.M_Iter:
            # same MOP and MOA formulas as your reference
            MOP = 1 - ((C_Iter) ** (1.0 / self.Alpha) / (self.M_Iter ** (1.0 / self.Alpha)))
            MOA = self.MOP_Min + C_Iter * ((self.MOP_Max - self.MOP_Min) / self.M_Iter)

            improvements = 0
            for i in range(self.pop_size):
                for j in range(self.dim):
                    r1 = self.rng.random()

                    # handle scalar vs per-dimension LB/UB the same way your
                    # original code did (branch on np.size(LB) == 1)
                    if np.size(self.LB) == 1:
                        # scalar bounds
                        if r1 < MOA:
                            r2 = self.rng.random()
                            if r2 > 0.5:
                                Xnew[i, j] = self.Best_P[j] / (MOP + self.eps) * ((self.UB - self.LB) * self.Mu + self.LB)
                            else:
                                Xnew[i, j] = self.Best_P[j] * MOP * ((self.UB - self.LB) * self.Mu + self.LB)
                        else:
                            r3 = self.rng.random()
                            if r3 > 0.5:
                                Xnew[i, j] = self.Best_P[j] - MOP * ((self.UB - self.LB) * self.Mu + self.LB)
                            else:
                                Xnew[i, j] = self.Best_P[j] + MOP * ((self.UB - self.LB) * self.Mu + self.LB)

                    else:
                        # per-dimension bounds
                        if r1 < MOA:
                            r2 = self.rng.random()
                            if r2 > 0.5:
                                Xnew[i, j] = self.Best_P[j] / (MOP + self.eps) * ((self.UB[j] - self.LB[j]) * self.Mu + self.LB[j])
                            else:
                                Xnew[i, j] = self.Best_P[j] * MOP * ((self.UB[j] - self.LB[j]) * self.Mu + self.LB[j])
                        else:
                            r3 = self.rng.random()
                            if r3 > 0.5:
                                Xnew[i, j] = self.Best_P[j] - MOP * ((self.UB[j] - self.LB[j]) * self.Mu + self.LB[j])
                            else:
                                Xnew[i, j] = self.Best_P[j] + MOP * ((self.UB[j] - self.LB[j]) * self.Mu + self.LB[j])

                # bounds correction
                if np.size(self.LB) == 1:
                    Xnew[i, :] = np.clip(Xnew[i, :], self.LB, self.UB)
                else:
                    Xnew[i, :] = np.minimum(np.maximum(Xnew[i, :], self.LB), self.UB)

                # evaluation and greedy replacement
                self.Ffun_new[i] = self._eval(Xnew[i, :])
                if self.Ffun_new[i] < self.Ffun[i]:
                    X[i, :] = Xnew[i, :]
                    self.Ffun[i] = self.Ffun_new[i]
                    improvements += 1
                if self.Ffun[i] < self.Best_FF:
                    self.Best_FF = self.Ffun[i]
                    self.Best_P = np.copy(X[i, :])

            # store convergence
            self.Conv_curve[C_Iter - 1] = self.Best_FF

            if verbose and (C_Iter % 5 == 0):
                print(f"At iteration {C_Iter} the best solution fitness is {self.Best_FF}")

            if debug and (C_Iter % debug_every == 0):
                self._debug_state(C_Iter, X, Xnew, improvements)

            C_Iter += 1

        return self.Best_FF, self.Best_P, self.Conv_curve


