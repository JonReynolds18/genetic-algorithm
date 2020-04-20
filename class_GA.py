# Convert the functional programming GA and implement it as a class based GA
import numpy as np
import math
import time


# Main class that handles the whole order of operations
class GA:
    
    # Optional GA settings are introduced via kwargs
    # All params should be passed to the GA object and inherited down
    def __init__(self, number_of_vars, lower_bound, upper_bound, fitness_info, **kwargs):
        self.number_of_vars = number_of_vars
        self.lower_bound = lower_bound
        self.upper_bound = upper_bound
        self.fitness_info = fitness_info
        self.population_size = kwargs.get('population_size', 100)
        self.number_of_generations = kwargs.get('number_of_generations', 100)
        self.elite_fraction = kwargs.get('elite_fraction', 0.05)
        self.crossover_fraction = kwargs.get('crossover_fraction', 0.8)
        self.decimal_places = kwargs.get('decimal_places', 1)
        self.selection_func = kwargs.get('selection_func', 'tournament_selection')
        self.selection_params = kwargs.get('selection', {'method': 'tournament_selection', 'param': 3})
        self.crossover_params = kwargs.get('crossover', {'method': 'crossover_scattered', 'param': None})
        self.mutation_params = kwargs.get('mutation', {'method': 'mutation_uniform', 'param': 0.1})
        self.diversity_limit = kwargs.get('diversity_limit', 0)
        self.creation_function_name = kwargs.get('custom_creation_function', 'default_creation_function')
        self.log = {'best_GA_fitness':0, 'best_GA_Individual':np.zeros((1,self.number_of_vars)), 'all_pops':np.zeros((self.population_size, self.number_of_vars, self.number_of_generations)),
                    'fittest_ind_per_pop':np.zeros((self.number_of_generations, self.number_of_vars)), 'best_fitness_per_pop': np.zeros((self.number_of_generations)), 'diversity': np.zeros((self.number_of_generations))}
        self.generation = 0
        self.output_location = kwargs.get('output_location', None)

        
    def run(self):
        
        orginal_mutation_rate = self.mutation_params['param']
        original_crossover_fraction = self.crossover_fraction
        
        while self.generation < self.number_of_generations:  
            
            if(self.generation+1) % 5 == 0:
                if self.output_location == None:
                    print("Progress = " + str(round((self.generation+1)/self.number_of_generations * 100)) + '%')
                else:
                    sheet = self.output_location[0]
                    sheet.range(self.output_location[1]).value = "Run Progress = " + str(round((self.generation+1)/self.number_of_generations * 100)) + '%'
                    
                    
            if self.generation == 0:
                self.population = Population(self.generation, self)
                
            else:
                # IMPLEMENT ADAPTIVE MUTATION RATES
                if(self.population.diversity< self.diversity_limit):
                    self.crossover_fraction = 0.5
                    self.mutation_params['param']=0.5
                else:
                    self.crossover_fraction = original_crossover_fraction
                    self.mutation_params['param']= orginal_mutation_rate
                    
                # SELECTION
                elite_parents = self.elite_function()
                selection_method = getattr(self, self.selection_params['method'])
                crossover_parents = selection_method(self.selection_params['param'], self.crossover_fraction)
                mutation_parents = selection_method(self.selection_params['param'], (self.population_size - len(elite_parents) - len(crossover_parents))/self.population_size)
                
                # CHILDREN
                crossover_method =  getattr(self, self.crossover_params['method'])
                mutation_method =  getattr(self, self.mutation_params['method'])
                crossover_children = crossover_method(crossover_parents)
                mutation_children = mutation_method(mutation_parents)
                
                # RECOMBINATION
                kwargs = {'elite_children':elite_parents, 'crossover_children':crossover_children, 'mutation_children':mutation_children}
                self.population = Population(self.generation, self, **kwargs)
            
            self.population.population_logs()
            self.generation+=1
        
    
    # Takes the best solutions from the old population proportional in size to the elite fraction
    def elite_function(self):
        number_of_parents = int(self.elite_fraction * self.population_size)
        ranked = self.population.rank_pop()
        parents = ranked[:number_of_parents]
        return parents
    
    # Takes 3 individuals at random from the old population, selects the best solution of these three
    # Does this to return a list of selected parents with length equal to the fraction (e.g. 80 for crossover on detail)
    def tournament_selection(self, tournament_size, fraction):
        number_of_parents = int(self.population_size*fraction)
        parents = []
        for ind in range(0,number_of_parents):
            selections = np.random.randint(0, high = self.population_size, size = tournament_size)
            selected_individual = self.population.population_obj[selections[0]]
            fittest = self.population.population_obj[selections[0]].fitness
            for i in range(1,tournament_size):
                if self.population.population_obj[selections[i]].fitness < fittest:
                    selected_individual = self.population.population_obj[selections[i]]
            parents.append(selected_individual)                 
        return parents
    
    def crossover_scattered(self, parents):
        crossover_children = []
        for i in range(0,math.ceil(len(parents)/2)):
            scattered_vector = np.random.randint(0,2,self.number_of_vars)
            inverse_vector = 1-scattered_vector
            child1_genes = scattered_vector*parents[i*2].genes + inverse_vector*parents[i*2+1].genes
            child2_genes = scattered_vector*parents[i*2+1].genes + inverse_vector*parents[i*2].genes
            
            crossover_children.append(self.manual_individual(child1_genes))
            crossover_children.append(self.manual_individual(child2_genes))
            
        return crossover_children
    
    def mutation_uniform(self, parents):
        mutation_children = []
        for i in range(0, math.ceil(len(parents))):
            mutation_vector = np.random.choice([0,1], self.number_of_vars, p=[1-self.mutation_params['param'], self.mutation_params['param']])
            inverse_vector = 1-mutation_vector
            mutations = Individual(self)
            mutation_child_genes = parents[i].genes*inverse_vector + mutations.genes*mutation_vector
            mutation_children.append(self.manual_individual(mutation_child_genes))
        
        return mutation_children
    
    def manual_individual(self, genes):
        child = Individual(self)
        child.genes = genes
        return child


# Class to create individual solutions, main atributes are the genes and the fitness
class Individual:
    
    def __init__(self, GA):
        # Pass the meta GA object to inherit all its attributes
        self.GA = GA
        
        if self.GA.creation_function_name == 'default_creation_function':
            self.default_creation_function()
        else:
            self.genes = self.GA.creation_function_name(self.GA)

        self.fitness = 0
    
    def default_creation_function(self):
        genes = np.random.rand(self.GA.number_of_vars)*(self.GA.upper_bound-self.GA.lower_bound) + self.GA.lower_bound
        self.genes = np.round(genes, self.GA.decimal_places)
        return self.genes
    




# Population is a class that holds the key information about each generation
class Population(GA):
    
    def __init__(self, generation, GA, **kwargs):
        # Pass the meta GA object to inherit all its attributes
        self.generation = generation
        self.GA = GA
        self.elite_children = kwargs.get('elite_children')
        self.crossover_children = kwargs.get('crossover_children')
        self.mutation_children = kwargs.get('mutation_children')
        
        
        # Calculating fitnesses when the whole population is formed rather than when each individual is formed
        if self.generation == 0:
            self.initialise_pop()
            # Loops through all individuals to calculate fitness
#            list(map(self.calc_fitness, self.population_obj))
            for individual in self.population_obj:
                self.calc_fitness(individual)            

        else:
            self.form_new_pop()
#            list(map(self.calc_fitness, self.population_obj))
            for individual in self.population_obj:
                self.calc_fitness(individual)
                
        self.diversity = self.calc_pop_diversity()

    def calc_fitness(self, individual):
        if len(self.GA.fitness_info)>1:
            individual.fitness = self.GA.fitness_info['fitness_function'](individual.genes, self.GA.fitness_info['additional_data'])
        else:
            individual.fitness = self.GA.fitness_info['fitness_function'](individual.genes)

    # Produces the population randomly during the first generation - population_obj is a list of Individual objects
    def initialise_pop(self):
        self.population_obj = [Individual(self.GA) for i in range(0,self.GA.population_size)]
    
    # Converts the population_obj list into a numpy array of the population and correspoding fitnesses
    def get_pop_summary(self, population_obj):
        pop = [obj.genes for obj in population_obj]
        fitnesses = [obj.fitness for obj in population_obj]

        population_array = np.array(pop)
        population_fitnesses = np.array(fitnesses)
        return population_array, population_fitnesses
    
    # Able to sort each population wrt their fitnesses - can be used for future selection methods   
    def rank_pop(self):
        self.ranked_pop = sorted(self.population_obj, key = lambda x: x.fitness, reverse=False)
        return self.ranked_pop
    
    # Form new populations for generation 1 onwards, receives children from crossover, mutation and elitism
    def form_new_pop(self):
        self.population_obj = self.elite_children + self.crossover_children + self.mutation_children
    
    # Calculate diveristy of solutions to allow adaptive mutation
    def calc_pop_diversity(self):
        distance = 0
        for individual in self.population_obj:
            distance = distance + np.linalg.norm(np.array(self.rank_pop()[0].genes - individual.genes), axis = 0)
        ave_distance = distance/self.GA.population_size
        return ave_distance
    
    # Keep a log of several useful results for debugging or analysis
    def population_logs(self):
        self.GA.log['best_GA_fitness']= self.rank_pop()[0].fitness
        self.GA.log['best_GA_Individual'] = self.rank_pop()[0].genes
        self.GA.log['all_pops'][:,:,self.GA.generation] = self.get_pop_summary(self.population_obj)[0]
        self.GA.log['fittest_ind_per_pop'][self.generation, :] = self.rank_pop()[0].genes
        self.GA.log['best_fitness_per_pop'][self.generation] = self.rank_pop()[0].fitness
        self.GA.log['diversity'][self.generation] = self.diversity

if __name__ == '__main__':
    
    # Simple optimisation problem as an example 
    start = time.time()
    
    def custom_creation(GA):
        genes = np.random.rand(GA.number_of_vars)*(GA.upper_bound-GA.lower_bound) + GA.lower_bound
        genes = np.round(genes, GA.decimal_places)
        return genes

    def fitness_function(genes):
        fitness = 0
        for i in range(0, len(genes)):
            fitness += math.fabs(genes[i] - i)
        return fitness
    
    optional_info = {'custom_creation_function': custom_creation, 'diversity_limit':0.5, 'population_size':100}
    fitness_info = {'fitness_function': fitness_function}
    ga = GA(number_of_vars=10, lower_bound=0, upper_bound=10, fitness_info=fitness_info, **optional_info)
    ga.run()
    results = ga.log
    end = time.time()
    time_taken = end-start



            
        
    
