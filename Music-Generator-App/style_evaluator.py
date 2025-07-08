#style_evaluator.py

import os
import glob
from collections import Counter
import numpy as np
from music21 import converter, note as m21_note, stream as m21_stream, duration as m21_duration, chord as m21_chord

# Importujemo konstante iz našeg config fajla
import config

# Šalje poruku u log red ili ispisuje na konzolu ako red nije dostupan
def log_message(message, text_widget_or_queue=None):
    if text_widget_or_queue:
        text_widget_or_queue.put(message + "\n")
    else:
        print(message)


# Klasa za učenje i ocjenjivanje muzičkog stila na osnovu MIDI fajlova
class StyleEvaluator:
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

    # Interna funkcija za logovanje poruka (koristi red ili konzolu)
    def _log(self, message):
        log_message(message, self.logger_queue)

    # Normalizuje brojanje vrijednosti u distribuciju vjerovatnoće
    def _normalize_counter(self, counts_counter, all_keys_for_distribution=None):
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

    # Kvantizuje (zaokružuje) trajanje note na najbližu dozvoljenu vrijednost
    def _quantize_duration(self, duration):
        bins = config.POSSIBLE_DURATIONS
        if not bins:
            return duration
        return min(bins, key=lambda x: abs(x - duration))

    # Izvlači muzičke karakteristike iz liste nota (tonovi, intervali, trajanja itd.)
    def _extract_features(self, melody_dict_list):
        pitches_for_style = [note['pitch'] for note in melody_dict_list if note['pitch'] is not None]
        pitch_classes_counts, intervals_counts, pitch_class_bigrams_counts = Counter(), Counter(), Counter()

        if pitches_for_style:
            for p in pitches_for_style:
                pitch_classes_counts[p % 12] += 1
            if len(pitches_for_style) > 1:
                for i in range(len(pitches_for_style) - 1):
                    p1, p2 = pitches_for_style[i], pitches_for_style[i + 1]
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

    # Uči muzički stil analizirajući skup MIDI fajlova u datom folderu
    def learn_style_from_dataset(self, dataset_folder_path):
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
            self._log("Nijedan MIDI fajl nije uspješno obrađen. Učenje stila neuspješno.")
            return False

        self.style_pitch_class_dist = self._normalize_counter(corpus_pc_counts, self._all_pitch_classes)
        self.style_interval_dist = self._normalize_counter(corpus_interval_counts, self._all_intervals)
        self.style_pitch_class_bigram_dist = self._normalize_counter(corpus_bigram_counts)
        self.style_duration_dist = self._normalize_counter(corpus_duration_counts, self._all_durations_for_dist)
        self.style_ioi_dist = self._normalize_counter(corpus_ioi_counts, self._all_iois_for_dist)
        self._log(f"Učenje stila završeno. Obrađeno {processed_files} fajlova.")
        return True

    # Izračunava Bhattacharyya koeficijent između dvije distribucije
    def _calculate_bhattacharyya_coefficient(self, dist1_dict, dist2_dict, all_keys):
        bc = sum(np.sqrt(dist1_dict.get(key, 0.0) * dist2_dict.get(key, 0.0)) for key in all_keys)
        return bc

    # Procjenjuje koliko je data melodija slična prethodno naučenom stilu
    def calculate_fitness(self, melody_dict_list):
        if not all([self.style_pitch_class_dist, self.style_interval_dist,
                    self.style_pitch_class_bigram_dist, self.style_duration_dist, self.style_ioi_dist]):
            return 0.0
        if not melody_dict_list:
            return 0.0

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
