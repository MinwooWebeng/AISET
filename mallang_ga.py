import numpy as np
import random
import soundfile as sf
import os
from concurrent.futures import ThreadPoolExecutor

import make_audio as maudio
import fitness

GENOME_LENGTH = 42
POPULATION = 100
ROYAL_SET = 10
TOURNAMENT_SIZE = 5

MAX_GENERATIONS = 100
MUTATION_RATE = 0.8
GENE_MUTATION_RATE = 0.2
GENE_MUTATION_STRENGTH = 0.05

INPUT_AUDIO = "./10s_datasets/clean/audio_Ballade_4.wav"
TARGET_AUDIO = "./10s_datasets/target.wav"

i_name = 0
class Individual:
    def __init__(self, gen=[]):
        global i_name
        if len(gen) == 0:
            self.genome = np.random.uniform(low=0.0, high=1.0, size=GENOME_LENGTH)
        else:
            self.genome = gen
        self.fitness = None
        self.i_name = i_name
        i_name += 1

    def evaluate(self):
        processed_audio, sr = maudio.FX_to_Audio(self.genome, INPUT_AUDIO)
        sf.write("temp/" + str(self.i_name) + "_temp_audio.wav", processed_audio, sr)
        fq_fitness, _, _, _ = fitness.fitness_frequency_domain("temp/" + str(self.i_name) + "_temp_audio.wav", TARGET_AUDIO)
        tm_fitness, _, _, _ = fitness.fitness_time_domain("temp/" + str(self.i_name) + "_temp_audio.wav", TARGET_AUDIO)
        self.fitness = 0.5 * fq_fitness + 0.5 - 5 * (1 - tm_fitness)
        os.remove("temp/" + str(self.i_name) + "_temp_audio.wav")
        
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
        # count = 1
        # for individual in self.people:
        #     print(f"evaluate({count}/{POPULATION})     ", end='\r')
        #     individual.evaluate()
        #     count += 1

        with ThreadPoolExecutor() as executor:
            executor.map(lambda x: x.evaluate(), self.people)

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
    print(f"Fitness = {str(generation.people[0].fitness)[:10]}, Value = {generation.people[0].genome}")

    for i in range(MAX_GENERATIONS):
        new_generation = Generation(is_rand=False)

        for j in range(ROYAL_SET):
            new_generation.people.append(generation.people[j])

        while len(new_generation.people) < POPULATION:
            child = crossover(generation.tournament_select(), generation.tournament_select())
            if random.random() < MUTATION_RATE:
                child.mutate()

            new_generation.people.append(child)

        new_generation.evaluate()
        print(f"Fitness = {str(new_generation.people[0].fitness)[:10]}, Value = {new_generation.people[0].genome}")
        processed_audio, sr = maudio.FX_to_Audio(new_generation.people[0].genome, INPUT_AUDIO)
        sf.write("MT_" + str(i) + "_best.wav", processed_audio, sr)

        generation = new_generation

if __name__ == "__main__":
    genetic_algorithm()
