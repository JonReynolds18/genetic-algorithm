# All GA functions contained in one Python script
import numpy as np
import math
import matplotlib.pyplot as plt

# This is the macro-level main function that controls the entire GA procedure
def myGA(fitness_info, number_of_vars, lower_bounds, upper_bounds, population_size = 100,
         number_of_generations = 1000, crossover_fraction = 0.8, elite_fraction = 0.05, decimal_places = 2,
         mutation_rate = 0.1, low_diversity = 0, plotting=False):

    if plotting == True:
        plt.ion()
        fig = plt.figure()
        plt.xlabel('Generation')
        plt.ylabel('Fitness')


    #----------------INITIALISATION----------------------
    #Create an initial population randomly
    population = initial_pop(number_of_vars, population_size, lower_bounds, upper_bounds, decimal_places)

    for generation in range(0,number_of_generations):

        #----------------FITNESS CALCULATION-----------------
        #Calculate the fitness of each individual
        population_fitness = []
        for individual in population:
        #The fitness function must be passed as a list object, if additional data
        #is required to calculate fitness, pass as the second object in the list
        #e.g. fitness_info = [fitness_function, additional_data]
            if len(fitness_info) == 1:
                population_fitness.append(fitness_info[0](individual))
            else:
                population_fitness.append(fitness_info[0](individual, fitness_info[1]))

        #--------------CALCULATE POP DIVERSITY---------------
        distance = 0
        for individual in population:
            #Sums the euclidian distance between the fittest and each individual
            distance = distance + np.linalg.norm(population[np.argmin(population_fitness)] - individual, axis = 0)
        ave_distance = distance/population_size

        #----------------ADAPTIVE MUTATION-------------------
        if ave_distance < low_diversity: #If pop diversity is low, throw a mutation bomb
            current_crossover_fraction = 0.5
            current_mutation_rate = 0.5
        else:
            current_crossover_fraction = crossover_fraction
            current_mutation_rate = mutation_rate

        #-----------------SELECTION--------------------------
        #Initialise parent populations
        elite_parents = elitism(population, population_fitness, elite_fraction)
        crossover_parents = np.zeros((math.ceil(population_size*current_crossover_fraction), number_of_vars))
        mutation_parents = np.zeros((population_size - elite_parents.shape[0] - crossover_parents.shape[0] ,number_of_vars))

        #Select population of parents designated for crossover using selection function
        for i in range(0, crossover_parents.shape[0]):
            crossover_parents[i,:] = tournament_selection(population, population_fitness)

        #Select population of parents designated for mutation using selection function
        for i in range(0, mutation_parents.shape[0]):
            mutation_parents[i,:] = tournament_selection(population, population_fitness)

        #----------------CROSSOVER---------------------------
        crossover_children = scattered_crossover(crossover_parents)

        #----------------MUTATION----------------------------
        mutation_children = mutation(mutation_parents, create_individual, lower_bounds, upper_bounds, decimal_places, mutation_rate= current_mutation_rate)

        #---------------SAVE OLD POP-------------------------
        if generation == 0:
            old_pops = population # Saves the entire population for debugging
            fittest_per_pop = population[np.argmin(population_fitness)] # Gives the fittest individual per population
            per_pop_fitness = min(population_fitness) # Gives the fitness value of the fittest individual per population
            print('Generation ' + str(generation+1) + ' completed, best fitness - ' + str(per_pop_fitness))
            diversity = ave_distance
        else:
            old_pops = np.dstack((old_pops, population))
            fittest_per_pop = np.vstack((fittest_per_pop, population[np.argmin(population_fitness)]))
            per_pop_fitness = np.vstack((per_pop_fitness, min(population_fitness)))
            print('Generation ' + str(generation+1) + ' completed, best fitness - ' + str(np.around(np.asscalar(per_pop_fitness[-1]),decimal_places)))
            diversity = np.vstack((diversity, ave_distance))

        #---------------PLOTTING----------------------------

        if plotting == True:
            fittest_plot = plt.scatter(generation, min(population_fitness), marker='.', c='black')
            diversity_plot = plt.scatter(generation, ave_distance, marker='.', c='blue')
            plt.legend((fittest_plot, diversity_plot), ('Fittest in Generation', 'Generation Diversity'))
            plt.show()
            plt.pause(0.000001)

        #---------------FORM NEXT GENERATION-----------------
        population = np.vstack((elite_parents, crossover_children, mutation_children))

    # Produce the final results once all generations have been completed
    optimal_value = per_pop_fitness[-1]
    optimal_solution = fittest_per_pop[-1,:]

#    plt.text(number_of_generations/3,per_pop_fitness.max() - 0.3*(per_pop_fitness.max() - per_pop_fitness.min()),'Final Fitness = ' + str(optimal_value))

    return optimal_value, optimal_solution, old_pops, fittest_per_pop, per_pop_fitness, diversity

# Default method to create an individual
def create_individual(number_of_vars, lower_bounds, upper_bounds, decimal_places):
    #Matrix of random floats between the lower and upper bounds
    individual = np.random.rand(1, number_of_vars)
    individual = individual * (upper_bounds - lower_bounds) + lower_bounds
    individual = np.around(individual, decimal_places)

    return individual

# Create the initial population randomly - (Might be redundant, just do this in main)
def initial_pop(number_of_vars, population_size, lower_bounds, upper_bounds, decimal_places, creation_function = create_individual):

    population = np.zeros((population_size,number_of_vars))
    for i in range(0, population_size):
        population[i] = creation_function(number_of_vars, lower_bounds, upper_bounds, decimal_places)

    return population

# Tournament selection process
def tournament_selection(population, population_fitness, tournament_size = 3):
    #Randomly pick tournament size worth of individuals from the population
    selections = np.random.randint(0, high = population.shape[0], size = tournament_size)
    selected_individual = population[selections[0]]
    fittest = population_fitness[selections[0]]

    #Compare randomly selected individuals, the lowest fitness is selected, NOTE:Low fitness is the best
    for i in range(1,tournament_size):
        if population_fitness[selections[i]] < fittest:
            selected_individual = population[selections[i]]

    return selected_individual

# Elitism function to select the best solutions from the population
def elitism(population, population_fitness, elite_fraction):
    elite_count = math.ceil(population.shape[0]*elite_fraction) #Number of elite parents
    inds = np.argpartition(population_fitness, elite_count)[:elite_count] #Get indices of parents with lowest(best) fitness

    elite_parents = np.zeros((elite_count, population.shape[1]))

    for i in range(0,elite_count):
        elite_parents[i,:] = population[inds[i],:]

    return elite_parents

def scattered_crossover(crossover_parents):
    crossover_children = np.zeros(crossover_parents.shape)
    
    for i in range(0,math.ceil(crossover_children.shape[0]/2)):
        #Create random binary vector of length = number of vars
        scattered_vector = np.random.randint(0,2,crossover_children.shape[1])
        #Create the inverse of that vector (0 to 1 and 1 to 0)
        inverse_vector = 1-scattered_vector
        #Recombine 2 parents based on the scattered vectors
        crossover_children[i*2,:] = scattered_vector*crossover_parents[i*2,:] + inverse_vector*crossover_parents[i*2+1, :]
        crossover_children[i*2+1, :] = scattered_vector*crossover_parents[i*2+1, :] + inverse_vector*crossover_parents[i*2, :]

    return crossover_children


def mutation(mutation_parents, creation_function, lower_bounds, upper_bounds, decimal_places, mutation_rate=0.1):
    mutation_children = np.zeros(mutation_parents.shape)

    for i in range(0, mutation_parents.shape[0]):
        # Binary vector with set chance of being a 1, these genes mutate
        mutation_vector = np.random.choice([0, 1], mutation_parents.shape[1], p=[1-mutation_rate, mutation_rate])
        inverse_vector = 1-mutation_vector
        mutations = creation_function(mutation_parents.shape[1], lower_bounds, upper_bounds, decimal_places)
        mutation_children[i, :] = mutation_parents[i, :]*inverse_vector + mutations*mutation_vector

    return mutation_children
