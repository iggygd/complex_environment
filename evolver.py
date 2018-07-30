import numpy as np
import random

def random_indices(array):
    return np.random.choice(len(array), np.random.choice(len(array)))

def evolve_flip(weights, *args):
    for i in random_indices(weights):
        for j in random_indices(weights[i]):
            weights[i][j] = -1*weights[i][j]
    return weights

def evolve_rand(weights, *args):
    for i in random_indices(weights):
        for j in random_indices(weights[i]):
            weights[i][j] = random.random()
    return weights

def evolve_add(weights, *args):
    for i in random_indices(weights):
        for j in random_indices(weights[i]):
            weights[i][j] = weights[i][j] + random.uniform(-1,1)
    return weights

def evolve_prog(weights, *args):
    for i in random_indices(weights):
        for j in random_indices(weights[i]):
            weights[i][j] = weights[i][j] * random.uniform(0.5,2)
    return weights

def evolve_merg(weights, b2, *args):
    for i in random_indices(weights):
        for j in random_indices(weights[i]):
            weights[i][j] = b2[i][j]
    return weights

def evolve_shuf(weights, *args):
    for i in random_indices(weights):
        random.shuffle(weights[i])
    return weights

def evolve(weights, b2 = None):
    if len(weights) is 0:
        return weights

    evolve_funcs = [evolve_flip, evolve_rand, evolve_add, evolve_prog, evolve_merg, evolve_shuf]
    if b2:
        pass
    else:
        evolve_funcs.remove(evolve_merg)

    weights = random.choice(evolve_funcs)(weights, b2)
    return weights
