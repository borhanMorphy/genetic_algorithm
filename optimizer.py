import random as rd
from neuralnet import NN, parse_input_stream
import functools
import numpy as np

class NNGeneticOptimizer:
    
    def __init__(self, population=100, mutation_chance=0.3, breed_candidate_count=2, max_generation=1000, **NNShape):
        self.mutation_chance = mutation_chance
        self.breed_candidate_count = breed_candidate_count
        # creating the population
        self.NNPopulation = [NN(**NNShape) for i in range(population)]
        self.generation = 1
        self.max_generation = max_generation
    
    def render(self,action):
        for i in range(len(self.NNPopulation)):
            for j in range(len(self.NNPopulation[i].layers)):
                print("%d NN  %d layer %s:\n"%(i,j,action),self.NNPopulation[i].layers[j].W,"\n")
                return
        print("--------------------------------")

    """
        calculate fitness on the population
    """
    def fitness(self):
        pass

    """
        select breed candidates from population
        fitness results as:
        [{'id':individual_index, 'fitness':point}]
    """
    def selection(self, fitness_results):
        return sorted(fitness_results, key=lambda i:i["fitness"], reverse=True)[:self.breed_candidate_count]

    """
        cross-over operation between breed candidates
        params: cadidates indexes as list
    """
    def cross_over(self, breed_candidates):
        # Populating next generation
        # Individual Iteration
        
        for NNIndividual in self.NNPopulation:
            # Layer iteration
            for layer_index,layer in enumerate(NNIndividual.layers):
                row, column = layer.W.shape
                for i in range(row):
                    for j in range(column):
                        index = rd.choice(breed_candidates)
                        layer.W[i][j] = self.NNPopulation[index].layers[layer_index].W[i][j]  # TODO: find a better way...
            # Mutation
            mutation = np.random.choice([True,False], size=1, p=[self.mutation_chance, 1-self.mutation_chance])[0]
            if mutation:
                self.mutate(NNIndividual)

    """
        mutate the next generation
    """
    def mutate(self, network_to_mutate):
        network = rd.choice(self.NNPopulation)
        layer_index = rd.randint(0, len(network.layers)-1)
        i = rd.randint(0, network.layers[layer_index].W.shape[0]-1)
        j = rd.randint(0, network.layers[layer_index].W.shape[1]-1)
        network_to_mutate.layers[layer_index].W[i][j] = network.layers[layer_index].W[i][j]
        return network_to_mutate
    


if __name__ == '__main__':
    nn_go = NNGeneticOptimizer(
        population=30,
        input_shape=(4,2), dense_shapes=[(2,5),(5,5)], output_shape=(5,3),
        mutation_chance=0.2
    )

    nn_go.cross_over([2,0])