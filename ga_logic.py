# ga_logic.py
import random
from config import MIN_PITCH_GA, MAX_PITCH_GA, POSSIBLE_DURATIONS, DEFAULT_VELOCITY

def create_random_note_for_ga():
    """Kreira jednu nasumičnu notu kao rečnik."""
    pitch = random.randint(MIN_PITCH_GA, MAX_PITCH_GA)
    duration = random.choice(POSSIBLE_DURATIONS)
    return {'pitch': pitch, 'duration': duration, 'velocity': DEFAULT_VELOCITY}

def create_random_melody_for_ga(length):
    """Kreira nasumičnu melodiju (listu nota) zadate dužine."""
    return [create_random_note_for_ga() for _ in range(length)]

def initialize_population_for_ga(pop_size, melody_length):
    """Inicijalizuje početnu populaciju melodija."""
    return [create_random_melody_for_ga(melody_length) for _ in range(pop_size)]

def selection_tournament(population, fitness_scores, tournament_size=3):
    """Bira roditelje koristeći turnirsku selekciju."""
    selected_parents = []
    pop_indices = list(range(len(population)))
    num_parents_to_select = len(population)

    actual_tournament_size = min(tournament_size, len(pop_indices))
    if actual_tournament_size == 0:
        return []

    for _ in range(num_parents_to_select):
        current_sample_size = min(actual_tournament_size, len(pop_indices))
        if current_sample_size == 0: break

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

def crossover_one_point(parent1_melody, parent2_melody, crossover_rate, melody_len):
    """Vrši ukrštanje u jednoj tački između dva roditelja."""
    if random.random() < crossover_rate and melody_len > 1:
        point = random.randint(1, melody_len - 1)
        child1_melody = parent1_melody[:point] + parent2_melody[point:]
        child2_melody = parent2_melody[:point] + parent1_melody[point:]
        return child1_melody, child2_melody
    return parent1_melody[:], parent2_melody[:]

def mutate_pitch_duration_for_ga(melody, mutation_rate):
    """Mutira note u melodiji (menja visinu ili trajanje)."""
    mutated_melody = []
    for note_dict in melody:
        new_note_dict = note_dict.copy()
        if random.random() < mutation_rate:
            if random.random() < 0.7:  # Veća šansa da se mutira visina
                new_note_dict['pitch'] = random.randint(MIN_PITCH_GA, MAX_PITCH_GA)
            else:
                new_note_dict['duration'] = random.choice(POSSIBLE_DURATIONS)
        mutated_melody.append(new_note_dict)
    return mutated_melody