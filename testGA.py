# To test the components of the GA from outside the methods
import numpy as np
from completeGA import myGA
from fitness import my_fitness

# Import the GA and fitness function, create the required parameters and set the desired GA options

number_of_vars = 10
lower_bounds = np.array([-10,-10,-10,-10,-10,-10,-10,-10,-10,-10])
upper_bounds = np.array([10,10,10,10,10,10,10,10,10,10])

fitness_info = [my_fitness]

outcome = myGA(fitness_info, number_of_vars, lower_bounds, upper_bounds, 
               number_of_generations = 100, population_size= 100, elite_fraction= 0.01,
               crossover_fraction= 0.8, mutation_rate=0.1, low_diversity=2)

optimal = outcome[1]
