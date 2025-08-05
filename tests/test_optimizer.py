import numpy as np
from aoa_metaheuristic.optimizer import AOA

#Simple fitness func. for testing
def sphere(x):
    return np.sum(x**2)

def test_initialization():
    bounds = [(-5, 5)] * 3
    aoa = AOA(fitness_func=sphere, bounds=bounds, dim=3, pop_size=10, max_iter=10)
    
    # Verifica shape da população
    assert aoa.population.shape == (10, 3)
    
    # Verifica limites da população
    assert np.all(aoa.population >= -5)
    assert np.all(aoa.population <= 5)
    
    # Verifica melhor fitness inicial
    assert isinstance(aoa.best_fitness, float)

def test_update_position_within_bounds():
    bounds = [(0, 10)] * 2
    aoa = AOA(fitness_func=sphere, bounds=bounds, dim=2, pop_size=5, max_iter=10)
    
    # Fixar valores para teste determinísticoc
    np.random.seed(0)
    current = np.array([5.0, 5.0])
    best = np.array([7.0, 8.0])
    moa = 0.5
    mop = 0.5
    
    new_val0 = aoa._update_position(current, best, 0, moa, mop)
    new_val1 = aoa._update_position(current, best, 1, moa, mop)
    
    # Verifica se novos valores estão dentro dos limites
    assert 0 <= new_val0 <= 10
    assert 0 <= new_val1 <= 10

def test_optimize_improves_solution():
    bounds = [(-10, 10)] * 2
    aoa = AOA(fitness_func=sphere, bounds=bounds, dim=2, pop_size=20, max_iter=50)
    
    best_sol, best_fit = aoa.optimize(verbose=False)
    
    # Testa se o fitness é razoavelmente baixo para o problema Sphere (mínimo = 0)
    assert best_fit < 1.0
    
    # Verifica se solução está dentro dos limites
    for i, (lb, ub) in enumerate(bounds):
        assert lb <= best_sol[i] <= ub