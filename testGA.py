# To test the components of the GA from outside the methods
import numpy as np
from completeGA import myGA
from fitness import my_fitness, beale_function, booth_function

number_of_vars = 10
population_size = 100
lower_bounds = np.array([-10,-10,-10,-10,-10,-10,-10,-10,-10,-10])
upper_bounds = np.array([10,10,10,10,10,10,10,10,10,10])

fitness_info = [my_fitness, additional_data]

outcome = myGA(fitness_info, number_of_vars, lower_bounds, upper_bounds, 
               number_of_generations = 100, population_size= 100, elite_fraction= 0.01,
               crossover_fraction= 0.8, mutation_rate=0.1, low_diversity=2)

#number_of_vars = 2
#lower_bounds = np.array([-4.5, -4.5])
#upper_bounds = np.array([4.5, 4.5])
#
#outcome = myGA(beale_function, number_of_vars, lower_bounds, upper_bounds, 
#               number_of_generations = 100, mutation_rate = 0.5, crossover_fraction=0.5)

#number_of_vars = 2
#lower_bounds = np.array([-10, -10])
#upper_bounds = np.array([10, 10])
#
#outcome = myGA(booth_function, number_of_vars, lower_bounds, upper_bounds, 
#               number_of_generations = 100, low_diversity=1)

optimal = outcome[1]