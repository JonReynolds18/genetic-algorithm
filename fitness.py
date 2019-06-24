import numpy as np
import math

def my_fitness(individual):
    
    fitness = 0
    for i in range(0, np.shape(individual)[0]):
        fitness = fitness + math.fabs(individual[i] - i)
    
    return fitness

def beale_function(individual):
    x = individual[0]
    y = individual[1]
    
    fitness = (1.5 -x + x*y)**2 + (2.25 - x + x*y**2)**2 + (2.625 - x + x*y**3)**2
    return fitness

def booth_function(individual):
    x = individual[0]
    y = individual[1]
    fitness = (x + 2*y - 7)**2 + (2*x + y - 5)**2
    
    return fitness