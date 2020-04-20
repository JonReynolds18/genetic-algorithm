# genetic-algorithm
A python version of a genetic algorithm with adaptive mutation

This repo contains the main script (class_GA.py) containing the GA classes, functions and all of the sub-functions it requires to run. A small example is included within the __main__ of this script which aims to produce a matrix as close to the target matrix (0,1,2,3,4,5,6,7,8,9) as possible.

NOTE to users - You  shouldn't need to understand or change the main GA script simply create an instance of the GA class, provide the required inputs and call the .run() method.

There are several required inputs:
- **number_of_vars** - this is the number of decision variables in your optimisation problem 
- **lower_bound** - this is the lowest value your decision variable can take (if you require more complex limits, pass information via this input to your custom creation function - see below)
- **upper_bound** - this is the largest value your decision variable can take (if you require more complex limits, pass information via this input to your custom creation function - see below)
- **fitness_info** - must be a dictionary of format 1.{"fitness_function": your_fitness_fuction} or 2.{"fitness_function": your_fitness_fuction, "additional_data": *extra_data*} - you **MUST** create a function that takes as an input 1. Just the genes of an individual as a numpy array, i.e. your decision variables or 2. The genes of an individual plus any extra data required to calculate its fitness passes in the *extra_data* variable. This function must have a **SINGLE** value as an output that describes the fitness of any solution judged against the optimisation objective.

Optional variables can be passed as a kwargs dictionary, they include:
- **population_size** - (default=100) - the number of individuals in each generation
- **number_of_generations** - (default=100) the maximum number of generations until the optimisation stops
- **elite_fraction** - (default=0.05) the fraction of the population that is designated elite (i.e. the best individuals directly copied into the next generation) NOTE - mutation_fraction = 1 - crossover_fraction - elite_fraction and is not directly inputted
- **crossover_fraction** - (default=0.8) the fraction of the popoulation that is selected to go through crossover method
- **decimal_places** - (default=1) number of decimal places each dicision variable can take, this is currently a catch all and not configurable to specific decision variables, use a custom creation function if this is required
- **selection_func** - (default='tournament_selection) tournament selection explained below, if you require a custom selection function create a custom function and pass it via this variable similar to the creation function shown in the example (input as 'custom_selection_func': your_func_name).

TODO
selection_params
crossover_params
mutation_params
diversity_limit
creation_function_name
output_location

Explain default methods - creation, selection, mutation
Add examples?



