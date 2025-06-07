# music_logic.py

import os
import sys
import random
import time
import glob
import pickle
import traceback
import wave
import multiprocessing
import subprocess
from collections import Counter

# Potrebne biblioteke - instalirati sa: pip install pretty_midi numpy pygame music21
import pretty_midi
import numpy as np
import pygame
from music21 import converter, note as m21_note, stream as m21_stream, duration as m21_duration, chord as m21_chord

# Importujemo konstante iz našeg config fajla
import config

def log_message(message, text_widget_or_queue=None):
    """Šalje poruku u red za logovanje ili je ispisuje na konzolu."""
    if text_widget_or_queue:
        # Pretpostavljamo da je text_widget_or_queue thread-safe Queue
        text_widget_or_queue.put(message + "\n")
    else:
        # Fallback na konzolu ako red nije dostupan
        print(message)

class StyleEvaluator:
    """Klasa za učenje i evaluaciju muzičkog stila iz MIDI fajlova."""
    def __init__(self, weights=None, max_interval_semitones=24, logger_queue=None):
        self.style_pitch_class_dist = None
        self.style_interval_dist = None
        self.style_pitch_class_bigram_dist = None
        self.style_duration_dist = None
        self.style_ioi_dist = None

        self.weights = weights if weights is not None else {
            "pitch_class_similarity": 0.25,
            "interval_similarity": 0.25,
            "bigram_avg_prob": 0.20,
            "duration_similarity": 0.15,
            "ioi_similarity": 0.15
        }
        self.max_interval_semitones = max_interval_semitones
        self._all_pitch_classes = list(range(12))
        self._all_intervals = list(range(-max_interval_semitones, max_interval_semitones + 1))
        self._all_durations_for_dist = sorted(list(set(config.POSSIBLE_DURATIONS)))
        self._all_iois_for_dist = sorted(list(set(config.POSSIBLE_DURATIONS)))
        self.logger_queue = logger_queue

    def _log(self, message):
        """Pomoćna funkcija za logovanje unutar klase."""
        log_message(message, self.logger_queue)

    def _normalize_counter(self, counts_counter, all_keys_for_distribution=None):
        """Normalizuje Counter objekat u distribuciju verovatnoće."""
        total_sum = sum(counts_counter.values())
        if total_sum == 0:
            if all_keys_for_distribution:
                return {k: 0.0 for k in all_keys_for_distribution}
            return {k: 0.0 for k in counts_counter}

        normalized_dist = {k: v / total_sum for k, v in counts_counter.items()}

        if all_keys_for_distribution:
            final_dist = {k: 0.0 for k in all_keys_for_distribution}
            final_dist.update(normalized_dist)
            return final_dist
        return normalized_dist

    def _quantize_duration(self, duration):
        """Kvantizuje trajanje note na najbližu vrednost iz POSSIBLE_DURATIONS."""
        bins = config.POSSIBLE_DURATIONS
        if not bins: return duration
        return min(bins, key=lambda x: abs(x - duration))

    def _extract_features(self, melody_dict_list):
        """Ekstrahuje muzičke karakteristike iz liste nota."""
        pitches_for_style = [note['pitch'] for note in melody_dict_list if note['pitch'] is not None]
        pitch_classes_counts, intervals_counts, pitch_class_bigrams_counts = Counter(), Counter(), Counter()

        if pitches_for_style:
            for p in pitches_for_style:
                pitch_classes_counts[p % 12] += 1
            if len(pitches_for_style) > 1:
                for i in range(len(pitches_for_style) - 1):
                    p1, p2 = pitches_for_style[i], pitches_for_style[i+1]
                    semitones = p2 - p1
                    if abs(semitones) <= self.max_interval_semitones:
                        intervals_counts[semitones] += 1
                    pc1, pc2 = p1 % 12, p2 % 12
                    pitch_class_bigrams_counts[(pc1, pc2)] += 1

        durations_counts, iois_counts = Counter(), Counter()
        for note_obj in melody_dict_list:
            quantized_dur = self._quantize_duration(float(note_obj['duration']))
            durations_counts[quantized_dur] += 1
            iois_counts[quantized_dur] += 1

        return pitch_classes_counts, intervals_counts, pitch_class_bigrams_counts, durations_counts, iois_counts

    def learn_style_from_dataset(self, dataset_folder_path):
        """Uči stil iz svih MIDI fajlova u datom folderu."""
        self._log(f"Učim stil iz MIDI fajlova u: {dataset_folder_path}")
        corpus_pc_counts, corpus_interval_counts, corpus_bigram_counts = Counter(), Counter(), Counter()
        corpus_duration_counts, corpus_ioi_counts = Counter(), Counter()

        midi_files = glob.glob(os.path.join(dataset_folder_path, "*.mid")) + \
                     glob.glob(os.path.join(dataset_folder_path, "*.midi"))

        if not midi_files:
            self._log("Greška: Nema MIDI fajlova u dataset folderu.")
            return False

        processed_files = 0
        for midi_file in midi_files:
            try:
                score = converter.parse(midi_file)
                current_file_melody_dicts = []
                # Uzimamo samo note i pauze iz prvog dela (part) ili spljoštenog fajla
                elements_to_parse = score.flat.notesAndRests
                for element in elements_to_parse:
                    pitch_val, duration_val = None, 0.0
                    if isinstance(element, m21_note.Note):
                        pitch_val, duration_val = element.pitch.midi, element.duration.quarterLength
                    elif isinstance(element, m21_note.Rest):
                        pitch_val, duration_val = None, element.duration.quarterLength
                    elif isinstance(element, m21_chord.Chord):
                        pitch_val = max(p.midi for p in element.pitches) if element.pitches else None
                        duration_val = element.duration.quarterLength
                    else:
                        continue
                    
                    current_file_melody_dicts.append({
                        'pitch': pitch_val,
                        'duration': self._quantize_duration(duration_val),
                        'velocity': config.DEFAULT_VELOCITY
                    })

                if current_file_melody_dicts:
                    pc_c, int_c, bigr_c, dur_c, ioi_c = self._extract_features(current_file_melody_dicts)
                    corpus_pc_counts.update(pc_c)
                    corpus_interval_counts.update(int_c)
                    corpus_bigram_counts.update(bigr_c)
                    corpus_duration_counts.update(dur_c)
                    corpus_ioi_counts.update(ioi_c)
                    processed_files += 1
            except Exception as e:
                self._log(f"    Greška pri obradi fajla {os.path.basename(midi_file)}: {e}")

        if processed_files == 0:
            self._log("Nijedan MIDI fajl nije uspešno obrađen. Učenje stila neuspešno.")
            return False

        self.style_pitch_class_dist = self._normalize_counter(corpus_pc_counts, self._all_pitch_classes)
        self.style_interval_dist = self._normalize_counter(corpus_interval_counts, self._all_intervals)
        self.style_pitch_class_bigram_dist = self._normalize_counter(corpus_bigram_counts)
        self.style_duration_dist = self._normalize_counter(corpus_duration_counts, self._all_durations_for_dist)
        self.style_ioi_dist = self._normalize_counter(corpus_ioi_counts, self._all_iois_for_dist)
        self._log(f"Učenje stila završeno. Obrađeno {processed_files} fajlova.")
        return True

    def _calculate_bhattacharyya_coefficient(self, dist1_dict, dist2_dict, all_keys):
        """Izračunava sličnost dve distribucije."""
        bc = sum(np.sqrt(dist1_dict.get(key, 0.0) * dist2_dict.get(key, 0.0)) for key in all_keys)
        return bc

    def calculate_fitness(self, melody_dict_list):
        """Izračunava fitness (ocenu) generisane melodije u odnosu na naučeni stil."""
        if not all([self.style_pitch_class_dist, self.style_interval_dist,
                    self.style_pitch_class_bigram_dist, self.style_duration_dist, self.style_ioi_dist]):
            return 0.0
        if not melody_dict_list: return 0.0

        gen_pc_c, gen_int_c, gen_bigram_c, gen_dur_c, gen_ioi_c = self._extract_features(melody_dict_list)

        current_pc_dist = self._normalize_counter(gen_pc_c, self._all_pitch_classes)
        current_interval_dist = self._normalize_counter(gen_int_c, self._all_intervals)
        current_duration_dist = self._normalize_counter(gen_dur_c, self._all_durations_for_dist)
        current_ioi_dist = self._normalize_counter(gen_ioi_c, self._all_iois_for_dist)

        pc_sim = self._calculate_bhattacharyya_coefficient(self.style_pitch_class_dist, current_pc_dist, self._all_pitch_classes)
        int_sim = self._calculate_bhattacharyya_coefficient(self.style_interval_dist, current_interval_dist, self._all_intervals)
        dur_sim = self._calculate_bhattacharyya_coefficient(self.style_duration_dist, current_duration_dist, self._all_durations_for_dist)
        ioi_sim = self._calculate_bhattacharyya_coefficient(self.style_ioi_dist, current_ioi_dist, self._all_iois_for_dist)

        bigram_avg_prob = 0.0
        num_bigrams_in_melody = sum(gen_bigram_c.values())
        if num_bigrams_in_melody > 0 and self.style_pitch_class_bigram_dist:
            total_bigram_log_prob = sum(
                count * np.log(self.style_pitch_class_bigram_dist.get(bigram, 1e-9))
                for bigram, count in gen_bigram_c.items()
            )
            bigram_avg_prob = np.exp(total_bigram_log_prob / num_bigrams_in_melody)

        fitness = (self.weights["pitch_class_similarity"] * pc_sim +
                   self.weights["interval_similarity"] * int_sim +
                   self.weights["duration_similarity"] * dur_sim +
                   self.weights["ioi_similarity"] * ioi_sim +
                   self.weights["bigram_avg_prob"] * bigram_avg_prob)

        return max(0.0, min(100.0, fitness * 100.0))

# --- GA i Pomoćne Funkcije ---

def create_random_note_for_ga():
    pitch = random.randint(config.MIN_PITCH_GA, config.MAX_PITCH_GA)
    duration = random.choice(config.POSSIBLE_DURATIONS)
    return {'pitch': pitch, 'duration': duration, 'velocity': config.DEFAULT_VELOCITY}

def create_random_melody_for_ga(length):
    return [create_random_note_for_ga() for _ in range(length)]

def initialize_population_for_ga(pop_size, melody_length):
    return [create_random_melody_for_ga(melody_length) for _ in range(pop_size)]

def selection_tournament(population, fitness_scores, tournament_size=3):
    selected_parents = []
    pop_indices = list(range(len(population)))
    for _ in range(len(population)):
        tournament_indices = random.sample(pop_indices, min(tournament_size, len(pop_indices)))
        best_contender_idx = max(tournament_indices, key=lambda idx: fitness_scores[idx])
        selected_parents.append(population[best_contender_idx])
    return selected_parents

def crossover_one_point(parent1, parent2, crossover_rate, melody_len):
    if random.random() < crossover_rate and melody_len > 1:
        point = random.randint(1, melody_len - 1)
        return parent1[:point] + parent2[point:], parent2[:point] + parent1[point:]
    return parent1[:], parent2[:]

def mutate_pitch_duration_for_ga(melody, mutation_rate):
    for i in range(len(melody)):
        if random.random() < mutation_rate:
            if random.random() < 0.7:  # Veća šansa da se menja visina
                melody[i]['pitch'] = random.randint(config.MIN_PITCH_GA, config.MAX_PITCH_GA)
            else:
                melody[i]['duration'] = random.choice(config.POSSIBLE_DURATIONS)
    return melody

def melody_dict_list_to_midi(melody_dicts, output_filename, instrument_name, bpm, logger_queue=None):
    try:
        midi_data = pretty_midi.PrettyMIDI(initial_tempo=float(bpm))
        instrument_program = pretty_midi.instrument_name_to_program(instrument_name)
        instrument = pretty_midi.Instrument(program=instrument_program)
        
        current_time = 0.0
        for note_data in melody_dicts:
            pitch = int(note_data['pitch'])
            duration_sec = float(note_data['duration']) * (60.0 / bpm)
            note_event = pretty_midi.Note(
                velocity=int(note_data['velocity']),
                pitch=pitch,
                start=current_time,
                end=current_time + duration_sec
            )
            instrument.notes.append(note_event)
            current_time += duration_sec
        
        midi_data.instruments.append(instrument)
        midi_data.write(output_filename)
        return output_filename
    except Exception as e:
        log_message(f"Greška pri pisanju MIDI fajla {output_filename}: {e}", logger_queue)
        return None

def convert_midi_to_wav(midi_file_path, wav_file_path, logger_queue=None):
    if not os.path.exists(config.SOUND_FONT_PATH):
        log_message(f"Greška: SoundFont fajl nije pronađen: {config.SOUND_FONT_PATH}", logger_queue)
        return None

    command = [
        'fluidsynth', '-ni', config.SOUND_FONT_PATH, midi_file_path,
        '-F', wav_file_path, '-r', '44100'
    ]
    try:
        # Sakrivanje prozora konzole na Windowsu
        startupinfo = None
        if os.name == 'nt':
            startupinfo = subprocess.STARTUPINFO()
            startupinfo.dwFlags |= subprocess.STARTF_USESHOWWINDOW
        
        subprocess.run(command, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, startupinfo=startupinfo)
        
        if os.path.exists(wav_file_path) and os.path.getsize(wav_file_path) > 0:
            return wav_file_path
        else:
            log_message("Greška: FluidSynth se završio bez greške, ali WAV fajl nije kreiran.", logger_queue)
            return None
    except FileNotFoundError:
        log_message("Greška: 'fluidsynth' nije pronađen. Proverite da li je instaliran i u sistemskom PATH-u.", logger_queue)
        return None
    except subprocess.CalledProcessError as e:
        log_message(f"Greška prilikom konverzije MIDI u WAV. stderr: {e.stderr.decode()}", logger_queue)
        return None
    except Exception as e:
        log_message(f"Neočekivana greška tokom WAV konverzije: {e}", logger_queue)
        return None

def play_audio_with_pygame(wav_path):
    """Pokreće reprodukciju WAV fajla u zasebnom procesu."""
    try:
        pygame.mixer.init(frequency=44100, size=-16, channels=2, buffer=2048)
        pygame.mixer.music.load(wav_path)
        pygame.mixer.music.play()
        while pygame.mixer.music.get_busy():
            time.sleep(0.1)
    except Exception as e:
        print(f"[Pygame Playback Process Error]: {e}")
    finally:
        pygame.mixer.quit()