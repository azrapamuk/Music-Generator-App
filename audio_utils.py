# audio_utils.py
import os
import subprocess
import time
import pygame
import pretty_midi

# Funkcija za logovanje (da ne zavisimo od UI komponente direktno)
def _log(message, logger_queue=None):
    if logger_queue:
        logger_queue.put(message + "\n")
    else:
        print(message)

def melody_dict_list_to_midi(melody_dicts, output_filename, instrument_name, bpm, logger_queue=None):
    """Konvertuje listu rečnika nota u MIDI fajl."""
    try:
        actual_bpm = float(bpm)
        if actual_bpm <= 0:
            actual_bpm = 120.0
            _log(f"Upozorenje: BPM vrednost {bpm} nije validna, korišćen podrazumevani 120 BPM.", logger_queue)
        midi_data = pretty_midi.PrettyMIDI(initial_tempo=actual_bpm)
    except ValueError:
        _log(f"Upozorenje: Nije moguće konvertovati BPM '{bpm}' u broj, korišćen podrazumevani 120 BPM.", logger_queue)
        midi_data = pretty_midi.PrettyMIDI(initial_tempo=120.0)

    try:
        instrument_program = pretty_midi.instrument_name_to_program(instrument_name)
    except ValueError:
        _log(f"Upozorenje: Nepoznato ime instrumenta '{instrument_name}'. Koristi se podrazumevani 'Acoustic Grand Piano'.", logger_queue)
        instrument_program = pretty_midi.instrument_name_to_program('Acoustic Grand Piano')

    instrument = pretty_midi.Instrument(program=instrument_program)
    current_time = 0.0
    for note_data in melody_dicts:
        pitch = int(note_data['pitch'])
        duration_beats = float(note_data['duration'])
        duration_sec = duration_beats * (60.0 / actual_bpm)
        velocity = int(note_data['velocity'])
        note_event = pretty_midi.Note(velocity=velocity, pitch=pitch, start=current_time, end=current_time + duration_sec)
        instrument.notes.append(note_event)
        current_time += duration_sec

    midi_data.instruments.append(instrument)
    try:
        midi_data.write(output_filename)
        return output_filename
    except Exception as e:
        _log(f"Greška pri pisanju MIDI fajla {output_filename}: {e}", logger_queue)
        return None

def convert_midi_to_wav(midi_file_path, wav_file_path, sound_font_sf2, logger_queue=None):
    """Konvertuje MIDI fajl u WAV koristeći FluidSynth."""
    if not os.path.exists(sound_font_sf2):
        _log(f"Greška: SoundFont fajl nije pronađen: {sound_font_sf2}", logger_queue)
        return None
    if not os.path.exists(midi_file_path):
        _log(f"Greška: MIDI fajl za konverziju nije pronađen: {midi_file_path}", logger_queue)
        return None

    fluidsynth_executable = "fluidsynth"
    command = [fluidsynth_executable, '-a', 'dummy', '-F', wav_file_path, '-r', '44100', '-g', '1.0', '-ni', sound_font_sf2, midi_file_path]
    
    process = None
    try:
        startupinfo = None
        if os.name == 'nt':
            startupinfo = subprocess.STARTUPINFO()
            startupinfo.dwFlags |= subprocess.STARTF_USESHOWWINDOW
            creationflags = subprocess.CREATE_NO_WINDOW
        else:
            creationflags = 0

        process = subprocess.Popen(
            command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, startupinfo=startupinfo, creationflags=creationflags
        )
        stdout_data, stderr_data = process.communicate(timeout=20)
        
        if process.returncode == 0:
            if os.path.exists(wav_file_path) and os.path.getsize(wav_file_path) > 0:
                return wav_file_path
            else:
                _log("Greška: FluidSynth se završio bez greške, ali WAV fajl nije kreiran.", logger_queue)
                if stderr_data: _log(f"FluidSynth stderr: {stderr_data.strip()}", logger_queue)
                return None
        else:
            _log(f"Greška prilikom konverzije MIDI u WAV. FluidSynth kod greške: {process.returncode}", logger_queue)
            if stderr_data: _log(f"FluidSynth stderr: {stderr_data.strip()}", logger_queue)
            return None
    except subprocess.TimeoutExpired:
        _log("Greška: FluidSynth je prekoračio vreme (20s) i zaglavio se.", logger_queue)
        if process: process.kill()
        return None
    except FileNotFoundError:
        _log(f"Greška: FluidSynth nije pronađen. Proverite da li je '{fluidsynth_executable}' instaliran i u PATH-u.", logger_queue)
        return None
    except Exception as e:
        _log(f"Neočekivana greška tokom WAV konverzije: {e}", logger_queue)
        if process and process.poll() is None: process.kill()
        return None

def play_audio_with_pygame(wav_path):
    """Robusna, izolovana funkcija za reprodukciju zvuka korišćenjem pygame.mixer."""
    clean_path = os.path.normpath(wav_path.strip())
    try:
        pygame.mixer.init(frequency=44100, size=-16, channels=2, buffer=2048)
        pygame.mixer.music.load(clean_path)
        pygame.mixer.music.play()
        while pygame.mixer.music.get_busy():
            time.sleep(0.1)
    except Exception as e:
        print(f"[Pygame Playback Process Error]: {e}")
    finally:
        pygame.mixer.quit()