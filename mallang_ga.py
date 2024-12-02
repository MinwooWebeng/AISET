import numpy as np
import random

GENOME_LENGTH = 4
POPULATION = 100
ROYAL_SET = 5
TOURNAMENT_SIZE = 5

MAX_GENERATIONS = 50
MUTATION_RATE = 0.8
GENE_MUTATION_RATE = 0.5
GENE_MUTATION_STRENGTH = 0.05

class Individual:
    def __init__(self, gen=[]):
        if len(gen) == 0:
            self.genome = np.random.uniform(low=0.0, high=1.0, size=GENOME_LENGTH)
        else:
            self.genome = gen
        self.fitness = None

    def evaluate(self):
        self.fitness = sum([0.25 - (x - 0.7) ** 2 for x in self.genome])
        
    def mutate(self):
        for i in range(len(self.genome)):
            if random.random() < GENE_MUTATION_RATE:
                # Gaussian mutation
                self.genome[i] += np.random.normal(0, GENE_MUTATION_STRENGTH)
                self.genome[i] = np.clip(self.genome[i], 0.0, 1.0)

class Generation:
    def __init__(self, is_rand=True):
        if is_rand:
            self.people = [Individual() for _ in range(POPULATION)]
        else:
            self.people = []

    def evaluate(self):
        for individual in self.people:
            individual.evaluate()
        self.people.sort(key=lambda x: x.fitness, reverse=True)

    def tournament_select(self) -> Individual:
        tournament = random.sample(self.people, TOURNAMENT_SIZE)
        tournament.sort(key=lambda ind: ind.fitness, reverse=True)
        return tournament[0]

def Regenerate(generation: Generation) -> Generation:
        new_population = []
    
def crossover(parent1: Individual, parent2: Individual) -> Individual:
    crossover_point = random.randint(1, GENOME_LENGTH - 1)
    return Individual(np.concatenate(
        [parent1.genome[:crossover_point], parent2.genome[crossover_point:]]
    ))

def genetic_algorithm():
    generation = Generation()
    generation.evaluate()

    for _ in range(MAX_GENERATIONS):
        new_generation = Generation(is_rand=False)

        for i in range(ROYAL_SET):
            new_generation.people.append(generation.people[i])

        while len(new_generation.people) < POPULATION:
            child = crossover(generation.tournament_select(), generation.tournament_select())
            if random.random() < MUTATION_RATE:
                child.mutate()

            new_generation.people.append(child)

        new_generation.evaluate()
        print(f"Fitness = {str(new_generation.people[0].fitness)[:10]}, Value = {new_generation.people[0].genome}")

        generation = new_generation

if __name__ == "__main__":
    genetic_algorithm()