import numpy as np
import random
import soundfile as sf
import os
from concurrent.futures import ThreadPoolExecutor

import make_audio as maudio
import fitness

GENOME_LENGTH = 48  # 유전자 (fp number) 갯수 
POPULATION = 100    # 한 세대 개체 수
ROYAL_SET = 5      # 한 세대에서 다음 세대로 복사할 상위 개체 수
TOURNAMENT_SIZE = 4 # Tournament selection의 selection set 크기

MAX_GENERATIONS = 100   # 최대 실행 세대
MUTATION_RATE = 0.9     # crossover 이후에 각각의 개체가 mutation을 겪을 확률
GENE_MUTATION_RATE = 0.1    # mutation을 겪는 개체에서, 각각의 유전자가 mutate 될 확률
GENE_MUTATION_STRENGTH = 0.05   # mutation 되는 유전자에서, gaussian distribution 으로 값을 +- 하는데, 그 distribution의 mean 값

INPUT_AUDIO = "./datasets/EGFxDataset/clean_neck/1-0.wav"   # 원본 오디오
TARGET_AUDIO = "./datasets/EGFxDataset/BD2_neck/1-0.wav"    # 변조된 오디오 (목표)

RESULT_FOLDER = "experiment_2"  # 각 세대별 최선을 모을 폴더

i_name = 0  # 개체 이름 (파일 생성시에 구분용)
class Individual:   # 유전자 (float array) 랑 fitness (float), 이름
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
        processed_audio, sr = maudio.FX_to_Audio(self.genome, INPUT_AUDIO)  # 현재 유전자 대로 오디오 변조
        sf.write("temp/" + str(self.i_name) + "_temp_audio.wav", processed_audio, sr)   # 변조된 파일 작성
        fq_fitness, _, _, _ = fitness.fitness_frequency_domain("temp/" + str(self.i_name) + "_temp_audio.wav", TARGET_AUDIO)    # frequency domain 유사도
        tm_fitness, _, _, _ = fitness.fitness_time_domain("temp/" + str(self.i_name) + "_temp_audio.wav", TARGET_AUDIO) # time domain 유사도
        self.fitness = 0.5 * fq_fitness + 0.5 - 5 * (1 - tm_fitness)    # 유사도 합치기 (비율 조정)
        os.remove("temp/" + str(self.i_name) + "_temp_audio.wav")   # 변조된 파일 삭제
        
    def mutate(self):
        for i in range(len(self.genome)):
            if random.random() < GENE_MUTATION_RATE:
                # Gaussian mutation
                self.genome[i] += np.random.normal(0, GENE_MUTATION_STRENGTH)
                self.genome[i] = np.clip(self.genome[i], 0.0, 1.0)  # 0.0 ~ 1.0 이내로 유전자 값 제한

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
            executor.map(lambda x: x.evaluate(), self.people)   # 각각 개체 evaluation 병렬 처리

        self.people.sort(key=lambda x: x.fitness, reverse=True) # fitness 높은것이 앞에 오게 정렬

    def tournament_select(self) -> Individual:  # crossover을 위한 parent 랜덤 고르기
        tournament = random.sample(self.people, TOURNAMENT_SIZE)
        tournament.sort(key=lambda ind: ind.fitness, reverse=True)
        return tournament[0]
    
def crossover(parent1: Individual, parent2: Individual) -> Individual:
    crossover_point = random.randint(1, GENOME_LENGTH - 1)
    return Individual(np.concatenate(
        [parent1.genome[:crossover_point], parent2.genome[crossover_point:]]    # 랜덤 지점 기준으로 앞/뒤 잘라서 합치기
    ))

def genetic_algorithm():
    generation = Generation()   # 랜덤 세대 생성
    generation.evaluate()
    print(f"Fitness = {str(generation.people[0].fitness)[:10]}, Value = {generation.people[0].genome}")

    for i in range(MAX_GENERATIONS):
        new_generation = Generation(is_rand=False)  # 빈 세대 생성

        for j in range(ROYAL_SET):
            new_generation.people.append(generation.people[j])  # 상위 개체들 그대로 복사

        while len(new_generation.people) < POPULATION:
            child = crossover(generation.tournament_select(), generation.tournament_select())   # crossover 로 개체 생성
            if random.random() < MUTATION_RATE: # mutation
                child.mutate()

            new_generation.people.append(child) # child 세대에 추가

        new_generation.evaluate()
        print(f"Fitness = {str(new_generation.people[0].fitness)[:10]}, Value = {new_generation.people[0].genome}")
        processed_audio, sr = maudio.FX_to_Audio(new_generation.people[0].genome, INPUT_AUDIO)
        sf.write("./" + RESULT_FOLDER + "/MT_" + str(i) + "_best.wav", processed_audio, sr)

        generation = new_generation # 다음 세대로

if __name__ == "__main__":
    genetic_algorithm()
