# genetic-algorithm
A python version of a genetic algorithm with adaptive mutation

This repo contains the main script (completeGA.py) containing the GA functions and all of the sub-functions it requires to run. In addition it contains application to very simple fitness function (fitness.py) which aims to produce a matrix as close to the target matrix (0,1,2,3,4,5 ...) as possible. A third script imports the ga function and the fitness function to run the optimisation.

NOTE to users - You  shouldn't need to understand or change the main GA script simply call the myGA function and provide the required inputs and tweak the parameters described below:

fitness_info - pass a list, the first object in the list is the name of the fitness function method, the second is optional to pass any additional data required to calculate the fitness

number_of_vars - simply an integer value representing the number of decision variables in your problem

lower_bounds - a numpy array with length equal to the number of variables, each value within the array provides a lower bound for each specific decision variable

upper_bounds - a numpy array with length equal to the number of variables, each value within the array provides an upper bound for each specific decision variable

population_size - (optional, default=100) the number of individuals in each population

number_of_generations - (optional, default=1000) the maximum number of generations until the optimisation stops

crossover_fraction - (optional, default=0.8) the fraction of the popoulation that is selected to go through crossover

elite_fraction - (optional, default=0.05) the fraction of the population that is designated elite (i.e. the best individuals directly copied into the next generation) NOTE - mutation_fraction = 1 - crossover_fraction - elite_fraction

decimal_places - (optional, default=2) number of decimal places each dicision variable can take, this is currently a catch all and not configurable to specific decision variables, use a custom creation function if this is required

mutation_rate - (optional, default=0.1) the liklihood a decision variable in a mutation child can mutate to a random feasible number 

low_diversity - (optional, default=0) the threashold for popoulation diversity to adaptively increase mutation rates. I suggest keeping this low for the first attempt, look at the diversity output, see what number it is converging at and adjust this parameter accordingly low diversity means the entire population is very similar, the default 0 will never be reached so by default there is no adaptive mutation

plotting - (optional, default=False) allows you to plot fitness vs generation to assess convergence, severly slows down the optimisation so I suggest using it initially to check everything is working and then turn it off


