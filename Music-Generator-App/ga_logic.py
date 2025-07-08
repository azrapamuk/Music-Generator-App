# ga_logic.py

import random
from config import MIN_PITCH_GA, MAX_PITCH_GA, POSSIBLE_DURATIONS, DEFAULT_VELOCITY

# Kreira jednu nasumičnu notu kao rječnik sa visinom, trajanjem i jačinom
def create_random_note_for_ga():
    pitch = random.randint(MIN_PITCH_GA, MAX_PITCH_GA)
    duration = random.choice(POSSIBLE_DURATIONS)
    return {'pitch': pitch, 'duration': duration, 'velocity': DEFAULT_VELOCITY}

# Kreira nasumičnu melodiju zadate dužine kao listu nota
def create_random_melody_for_ga(length):
    return [create_random_note_for_ga() for _ in range(length)]

# Inicijalizuje početnu populaciju nasumičnih melodija
def initialize_population_for_ga(pop_size, melody_length):
    return [create_random_melody_for_ga(melody_length) for _ in range(pop_size)]

# Bira roditelje iz populacije koristeći turnirsku selekciju (fitness se poredi u malim grupama)
def selection_tournament(population, fitness_scores, tournament_size=3):
    selected_parents = []
    pop_indices = list(range(len(population)))
    num_parents_to_select = len(population)

    actual_tournament_size = min(tournament_size, len(pop_indices))
    if actual_tournament_size == 0:
        return []

    for _ in range(num_parents_to_select):
        current_sample_size = min(actual_tournament_size, len(pop_indices))
        if current_sample_size == 0:
            break

        tournament_contenders_indices = random.sample(pop_indices, current_sample_size)

        best_contender_index_in_pop = -1
        best_fitness_in_tournament = -float('inf')

        for contender_idx_in_pop in tournament_contenders_indices:
            if fitness_scores[contender_idx_in_pop] > best_fitness_in_tournament:
                best_fitness_in_tournament = fitness_scores[contender_idx_in_pop]
                best_contender_index_in_pop = contender_idx_in_pop

        if best_contender_index_in_pop != -1:
            selected_parents.append(population[best_contender_index_in_pop])
    return selected_parents

# Ukršta dvije melodije u jednoj tački, razmjenjuju dijelove nakon te tačke
def crossover_one_point(parent1_melody, parent2_melody, crossover_rate, melody_len):
    if random.random() < crossover_rate and melody_len > 1:
        point = random.randint(1, melody_len - 1)
        child1_melody = parent1_melody[:point] + parent2_melody[point:]
        child2_melody = parent2_melody[:point] + parent1_melody[point:]
        return child1_melody, child2_melody
    return parent1_melody[:], parent2_melody[:]

# Mutira pojedine note u melodiji — može promijeniti visinu ili trajanje
def mutate_pitch_duration_for_ga(melody, mutation_rate):
    mutated_melody = []
    for note_dict in melody:
        new_note_dict = note_dict.copy()
        if random.random() < mutation_rate:
            if random.random() < 0.7:  
                new_note_dict['pitch'] = random.randint(MIN_PITCH_GA, MAX_PITCH_GA)
            else:
                new_note_dict['duration'] = random.choice(POSSIBLE_DURATIONS)
        mutated_melody.append(new_note_dict)
    return mutated_melody
