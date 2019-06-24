import numpy as np
import math

# In this simple optimisation, the algorithm aims to produce the target matrix which is [0,1,2,3,4... n] where n is the number of decision variables. The fitness of each solution (to be minimised) is the sum of the absolute error between target matrix and the individual. The absolute optimal is therefore 0.

def my_fitness(individual):
    
    fitness = 0
    for i in range(0, np.shape(individual)[0]):
        fitness = fitness + math.fabs(individual[i] - i)
    
    return fitness
