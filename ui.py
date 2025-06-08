#ui.py

import tkinter as tk
from tkinter import filedialog, messagebox
from tkinter import CENTER 
import ttkbootstrap as ttk
from ttkbootstrap.constants import *
import os
import sys
import time
import threading
import pickle
import traceback
import subprocess
import numpy as np
import random
import pygame

# Uvoz biblioteka za vizualizaciju zvučnog vala
from scipy.io import wavfile
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

# Uvozimo kod iz vlastitih modula
import config
from style_evaluator import StyleEvaluator
from ga_logic import (initialize_population_for_ga, selection_tournament,
                      crossover_one_point, mutate_pitch_duration_for_ga)
from audio_utils import (melody_dict_list_to_midi, convert_midi_to_wav)

# Glavna klasa aplikacije koja upravlja korisničkim interfejsom, logikom genetskog algoritma i audio obradom
class MusicGeneratorApp:

    # Inicijalizira glavnu klasu aplikacije.
    def __init__(self, root):
        self.root = root
        self.style = ttk.Style()
        self.root.title("Glazbeni Generator s genetskim algoritmom")
        self.root.geometry("1000x700")
        self.root.protocol("WM_DELETE_WINDOW", self._on_closing)
        pygame.mixer.init()

        # Inicijalizacija ključnih varijabli stanja aplikacije
        self.toast = None  # Objekt za prikazivanje notifikacija
        self.style_evaluator = None  # Objekt koji sadrži naučeni muzički stil
        self.can_run_ga_flag = False # Zastavica koja označava je li stilski model učitan
        self.current_best_melody_wav_path = None # Putanja do zadnje generirane .wav datoteke
        self.worker_thread = None # Referenca na pozadinsku nit (za GA ili učenje stila)
        self.stop_event = threading.Event() # Događaj za sigurno zaustavljanje pozadinske niti

        # Varijable za korisnički interfejs (UI), povezane s kontrolama
        self.instrument_var = tk.StringVar(value=config.GM_INSTRUMENTS[0])
        self.midi_folder_path_var = tk.StringVar(value=config.DRIVE_MIDI_FOLDER_PATH)
        self.population_size_var = tk.IntVar(value=config.GA_POPULATION_SIZE)
        self.generations_var = tk.IntVar(value=config.GA_NUM_GENERATIONS)
        self.melody_length_var = tk.IntVar(value=config.GA_MELODY_LENGTH)
        self.mutation_rate_var = tk.DoubleVar(value=config.GA_MUTATION_RATE * 100)
        self.crossover_rate_var = tk.DoubleVar(value=config.GA_CROSSOVER_RATE * 100)
        self.bpm_var = tk.IntVar(value=config.GA_BPM)

        # Atributi za vizualizator zvučnog vala
        self.fig = None
        self.ax = None
        self.plot_canvas_widget = None
        self.placeholder_label = None # Oznaka s porukom koja se prikazuje dok nema vizualizacije
        self.playhead_line = None # Vertikalna linija koja prati reprodukciju
        self.animation_job = None # ID posla za animaciju linije
        self.total_audio_duration = 0

        # Postavljanje korisničkog interfejsa i provjera potrebnih resursa
        self.setup_cool_ui()
        self.check_soundfont()
        self.show_toast("Aplikacija je pokrenuta", bootstyle=INFO)

        # Pokušaj automatskog učitavanja zadanog stilskog modela pri pokretanju
        if not self.try_import_default_style_model():
            if os.path.exists(self.midi_folder_path_var.get()):
                self.start_worker_thread(self.initialize_style_model, from_dataset=True)
            else:
                self.show_toast(f"Zadani direktorij sa skupom podataka nije pronađen.", bootstyle=WARNING)
                self.set_ui_state_ready("Skup podataka/Model nije pronađen.")

    # Osigurava sigurno zatvaranje aplikacije: zaustavlja reprodukciju, prekida pozadinske niti i oslobađa resurse
    def _on_closing(self):
        self.stop_playback()
        if self.worker_thread and self.worker_thread.is_alive():
            self.stop_event.set() 
        pygame.mixer.quit()
        self.root.destroy()

    # Konstruira glavni prozor aplikacije koristeći PanedWindow za podesivu podjelu interfejsa
    def setup_cool_ui(self):
        paned_window = ttk.PanedWindow(self.root, orient=HORIZONTAL)
        paned_window.pack(fill=BOTH, expand=True, padx=10, pady=10)
        
        sidebar_frame = ttk.Frame(paned_window, padding=15)
        paned_window.add(sidebar_frame, weight=1)
        self.create_sidebar_controls(sidebar_frame)
        
        main_content_frame = ttk.Frame(paned_window, padding=15)
        paned_window.add(main_content_frame, weight=3)
        self.create_main_content(main_content_frame)
        
        self.status_var = tk.StringVar(value="Spreman.")
        status_bar = ttk.Label(self.root, textvariable=self.status_var, relief=SUNKEN, anchor=W, padding=5)
        status_bar.pack(side=BOTTOM, fill=X)

    # Stvara kontrole u lijevom panelu za učitavanje modela i postavljanje GA parametara
    def create_sidebar_controls(self, parent):
        parent.grid_columnconfigure(0, weight=1)
        
        # Okvir za upravljanje stilskim modelom
        model_frame = ttk.LabelFrame(parent, text="1. Stilski Model", padding=10)
        model_frame.grid(row=0, column=0, sticky="ew", pady=(0, 20))
        model_frame.grid_columnconfigure(0, weight=1)
        ttk.Label(model_frame, text="Putanja do MIDI skupa podataka:").grid(row=0, column=0, columnspan=2, sticky="w")
        entry = ttk.Entry(model_frame, textvariable=self.midi_folder_path_var)
        entry.grid(row=1, column=0, sticky="ew", pady=(0,5))
        self.browse_midi_button = ttk.Button(model_frame, text="...", command=self.browse_midi_folder, bootstyle="secondary-outline", width=3)
        self.browse_midi_button.grid(row=1, column=1, padx=(5,0), pady=(0,5))
        self.load_midi_button = ttk.Button(model_frame, text="Učitaj stil iz skupa podataka", command=lambda: self.start_worker_thread(self.initialize_style_model, from_dataset=True), bootstyle="primary")
        self.load_midi_button.grid(row=2, column=0, columnspan=2, pady=(5,10), sticky="ew")
        self.import_style_button = ttk.Button(model_frame, text="Uvezi Model (.pkl)", command=self.import_style_model_dialog, bootstyle="info-outline")
        self.import_style_button.grid(row=3, column=0, columnspan=2, pady=2, sticky="ew")
        self.export_style_button = ttk.Button(model_frame, text="Izvezi Model (.pkl)", command=self.export_style_model_dialog, state=DISABLED, bootstyle="info-outline")
        self.export_style_button.grid(row=4, column=0, columnspan=2, pady=2, sticky="ew")

        # Okvir za postavljanje parametara Genetskog Algoritma
        ga_frame = ttk.LabelFrame(parent, text="2. GA Parametri", padding=10)
        ga_frame.grid(row=1, column=0, sticky="ew", pady=(0, 20))
        ga_frame.grid_columnconfigure(1, weight=1)
        param_entries = [("Populacija:", self.population_size_var, 10, 200), ("Generacije:", self.generations_var, 5, 200), ("Duljina melodije:", self.melody_length_var, 5, 100), ("BPM:", self.bpm_var, 30, 240)]
        rate_entries = [("Stopa mutacije:", self.mutation_rate_var, 0, 100), ("Stopa križanja:", self.crossover_rate_var, 0, 100)]
        for i, (text, var, p_from, p_to) in enumerate(param_entries):
            ttk.Label(ga_frame, text=text).grid(row=i, column=0, sticky=W, pady=3)
            scale = ttk.Scale(ga_frame, from_=p_from, to=p_to, variable=var); scale.grid(row=i, column=1, sticky="ew", padx=5)
            val_label = ttk.Label(ga_frame, width=4); val_label.grid(row=i, column=2, sticky=W)
            var.trace_add("write", lambda n,i,m,v=var,lbl=val_label: self._update_slider_label(v.get(), lbl, v)); self._update_slider_label(var.get(), val_label, var)
        for i, (text, var, p_from, p_to) in enumerate(rate_entries):
            row_idx = i + len(param_entries)
            ttk.Label(ga_frame, text=text).grid(row=row_idx, column=0, sticky=W, pady=3)
            scale = ttk.Scale(ga_frame, from_=p_from, to=p_to, variable=var); scale.grid(row=row_idx, column=1, sticky="ew", padx=5)
            val_label = ttk.Label(ga_frame, width=4); val_label.grid(row=row_idx, column=2, sticky=W)
            var.trace_add("write", lambda n,i,m,v=var,lbl=val_label: self._update_slider_label(v.get(), lbl, v)); self._update_slider_label(var.get(), val_label, var)

        # Okvir za odabir instrumenta
        instr_frame = ttk.LabelFrame(parent, text="3. Instrument", padding=10); instr_frame.grid(row=2, column=0, sticky="ew"); instr_frame.grid_columnconfigure(0, weight=1)
        instrument_combo = ttk.Combobox(instr_frame, textvariable=self.instrument_var, values=config.GM_INSTRUMENTS, state="readonly"); instrument_combo.grid(row=0, column=0, sticky="ew")

    # Stvara glavni dio interfejsa s vizualizatorom i glavnim akcijskim dugmadima
    def create_main_content(self, parent):
        parent.grid_rowconfigure(0, weight=1)
        parent.grid_rowconfigure(1, weight=0)
        parent.grid_columnconfigure(0, weight=1)

        # Okvir u kojem će biti smješten graf zvučnog vala
        visualizer_frame = ttk.LabelFrame(parent, text="Vizualizacija Zvučnog Vala", padding=5)
        visualizer_frame.grid(row=0, column=0, sticky="nsew", pady=(0, 10))
        visualizer_frame.grid_rowconfigure(0, weight=1)
        visualizer_frame.grid_columnconfigure(0, weight=1)

        # Inicijalizacija Matplotlib grafa za prikaz unutar Tkinter prozora
        bg_color = self.style.colors.get('bg')
        self.fig = Figure(figsize=(5, 2), dpi=100, facecolor=bg_color)
        self.ax = self.fig.add_subplot(111, facecolor=bg_color)
        
        self.ax.spines['top'].set_visible(False)
        self.ax.spines['right'].set_visible(False)
        self.ax.spines['left'].set_visible(False)
        self.ax.spines['bottom'].set_visible(False)
        self.ax.set_xticks([])
        self.ax.set_yticks([])
        self.fig.tight_layout()
        
        self.plot_canvas_widget = FigureCanvasTkAgg(self.fig, master=visualizer_frame)
        self.plot_canvas_widget.get_tk_widget().grid(row=0, column=0, sticky="nsew")

        # Postavljanje placeholder poruke u središte vizualizatora
        self.placeholder_label = ttk.Label(
            visualizer_frame,
            text="Vizualizacija će se pojaviti ovdje nakon generiranja melodije.",
            bootstyle="secondary",
            font=("Helvetica", 12)
        )
        self.placeholder_label.place(relx=0.5, rely=0.5, anchor=CENTER)

        # Okvir s glavnim dugmadima
        controls_frame = ttk.Frame(parent)
        controls_frame.grid(row=1, column=0, sticky="ew", pady=(10, 0))
        controls_frame.grid_columnconfigure(1, weight=1)

        self.run_ga_button = ttk.Button(controls_frame, text="Generiraj Melodiju", command=self.start_ga_worker, state=DISABLED, bootstyle="success, large")
        self.run_ga_button.grid(row=0, column=0, padx=(0, 20), ipady=10, ipadx=10)
        self.play_button = ttk.Button(controls_frame, text="Sviraj", command=self.play_last_melody, state=DISABLED, bootstyle="primary-outline")
        self.play_button.grid(row=0, column=2, padx=(0, 5))
        self.stop_button = ttk.Button(controls_frame, text="Zaustavi", command=self.stop_playback, state=DISABLED, bootstyle="danger-outline")
        self.stop_button.grid(row=0, column=3, padx=5)
        self.open_folder_button = ttk.Button(controls_frame, text="Otvori Direktorij", command=self.open_output_folder, bootstyle="secondary-outline")
        self.open_folder_button.grid(row=0, column=4, padx=5)
        self.progress_meter = ttk.Meter(controls_frame, metersize=180, padding=5, amountused=0, metertype="semi", subtext="Napredak GA", interactive=False, bootstyle='primary', textright='%')
        self.progress_meter.grid(row=0, column=5, sticky="e", padx=(20,0))
    
    # Učitava .wav datoteku i iscrtava njezin zvučni val na grafu
    def draw_waveform(self, wav_path):
        try:
            if self.placeholder_label:
                self.placeholder_label.place_forget()
                self.placeholder_label = None

            self.ax.clear()
            
            self.ax.spines['left'].set_visible(True)
            self.ax.spines['bottom'].set_visible(True)
            self.ax.spines['left'].set_color(self.style.colors.get('fg'))
            self.ax.spines['bottom'].set_color(self.style.colors.get('fg'))
            self.ax.tick_params(axis='x', colors=self.style.colors.get('fg'))
            self.ax.tick_params(axis='y', colors=self.style.colors.get('fg'))

            sample_rate, data = wavfile.read(wav_path)
            
            if data.ndim > 1:
                data = data[:, 0]
            
            self.total_audio_duration = len(data) / sample_rate
            time_axis = np.linspace(0, self.total_audio_duration, num=len(data))
            
            self.ax.plot(time_axis, data, color=self.style.colors.get('primary'))
            
            self.ax.axis('off')
            self.ax.set_xlim(0, self.total_audio_duration)
            self.fig.tight_layout()
            self.plot_canvas_widget.draw()
        except Exception as e:
            self.log_to_ui(f"Greška pri crtanju valnog oblika: {e}")
            self.show_toast("Nije moguće prikazati zvučni val.", bootstyle=DANGER)

    # Pokreće animaciju vertikalne linije (playhead) koja prati zvuk na grafiku
    def start_animation(self):
        if self.total_audio_duration == 0: return
        self.stop_animation() 
        start_time = time.time()
        self.playhead_line = self.ax.axvline(x=0, color=self.style.colors.get('danger'), lw=1.5)
        
        def update_playhead():
            elapsed_time = time.time() - start_time
            if elapsed_time > self.total_audio_duration:
                return 
            
            self.playhead_line.set_xdata([elapsed_time])
            self.plot_canvas_widget.draw_idle() 
            
            self.animation_job = self.root.after(10, update_playhead)
            
        update_playhead()

    # Zaustavlja animaciju i uklanja liniju s grafika
    def stop_animation(self):
        if self.animation_job:
            self.root.after_cancel(self.animation_job)
            self.animation_job = None
        if self.playhead_line:
            try:
                self.playhead_line.remove()
                self.playhead_line = None
                self.plot_canvas_widget.draw_idle()
            except (ValueError, AttributeError):
                self.playhead_line = None 

    # Prikazuje notifikaciju u donjem desnom uglu prozora
    def show_toast(self, message, bootstyle=DEFAULT, duration=4000):
        try:
            if self.toast and self.toast.winfo_exists(): self.toast.destroy()
        except tk.TclError: pass
        self.toast = tk.Toplevel(self.root)
        self.toast.wm_overrideredirect(True)
        self.toast.wm_transient(self.root)
        if sys.platform == "win32": self.toast.wm_attributes("-toolwindow", True)
        self.toast.wm_attributes("-topmost", True); self.toast.wm_attributes("-alpha", 0.9)
        toast_frame = ttk.Frame(self.toast, bootstyle=bootstyle, padding=10)
        toast_frame.pack(expand=True, fill=BOTH)
        label = ttk.Label(toast_frame, text=message, bootstyle=f"{bootstyle}-inverse")
        label.pack(expand=True, fill=BOTH, padx=10, pady=5)
        self.toast.update_idletasks()
        main_x, main_y, main_w, main_h = self.root.winfo_x(), self.root.winfo_y(), self.root.winfo_width(), self.root.winfo_height()
        toast_w, toast_h = self.toast.winfo_width(), self.toast.winfo_height()
        if main_w > 1 and main_h > 1:
            x, y = main_x + main_w - toast_w - 20, main_y + main_h - toast_h - 20
            self.toast.geometry(f"+{x}+{y}")
        else:
            self.toast.geometry(f"+{self.root.winfo_screenwidth() - toast_w - 20}+{self.root.winfo_screenheight() - toast_h - 100}")
        self.root.focus_force()
        self.toast.after(duration, lambda: self.toast.destroy())

    # Glavna petlja genetskog algoritma
    def run_ga_logic(self):
        try:
            # Dohvaćanje parametara iz korisničkog interfejsa
            num_generations = self.generations_var.get()
            self.root.after(0, lambda: self.progress_meter.configure(amountused=0, bootstyle='primary'))
            pop_size = self.population_size_var.get()
            
            population = initialize_population_for_ga(pop_size, self.melody_length_var.get())
            best_melody = None

            # Glavna petlja koja iterira kroz generacije
            for gen in range(num_generations):
                if self.stop_event.is_set(): 
                    self.show_toast("GA proces je prekinut.", bootstyle=WARNING)
                    break
                
                current_gen_num = gen + 1
                progress_percentage = (current_gen_num / num_generations) * 100
                self.root.after(0, lambda p=progress_percentage: self.progress_meter.configure(amountused=p))
                self.update_status_bar(f"Obrada generacije {current_gen_num} od {num_generations}...")

                fitness_scores = [self.style_evaluator.calculate_fitness(m) for m in population]
                if self.stop_event.is_set(): break
                
                best_idx = np.argmax(fitness_scores)
                best_melody = population[best_idx][:]
                
                parents = selection_tournament(population, fitness_scores)
                if not parents: break
                
                next_gen = [best_melody] 
                while len(next_gen) < pop_size:
                    p1, p2 = random.sample(parents, 2) if len(parents) >= 2 else (parents[0], parents[0])
                    c1, c2 = crossover_one_point(p1, p2, self.crossover_rate_var.get()/100.0, self.melody_length_var.get())
                    next_gen.append(mutate_pitch_duration_for_ga(c1, self.mutation_rate_var.get()/100.0))
                    if len(next_gen) < pop_size:
                        next_gen.append(mutate_pitch_duration_for_ga(c2, self.mutation_rate_var.get()/100.0))
                population = next_gen[:pop_size]

            # Nakon završetka svih generacija
            if not self.stop_event.is_set():
                self.root.after(0, lambda: self.progress_meter.configure(amountused=100))
                self.update_status_bar(f"GA završen nakon {num_generations} generacija.")
            else:
                self.update_status_bar(f"GA prekinut.")

            # Ako je pronađena najbolja melodija, sprema se kao MIDI i WAV datoteka
            if best_melody and not self.stop_event.is_set():
                self.update_status_bar("Generiram audio...")
                output_dir = os.path.join(config.BASE_DIR, config.OUTPUT_DIR_NAME)
                os.makedirs(output_dir, exist_ok=True)

                base_name = f"mel_{time.strftime('%Y%m%d-%H%M%S')}"
                midi_file = melody_dict_list_to_midi(best_melody, os.path.join(output_dir, f"{base_name}.mid"), self.instrument_var.get(), self.bpm_var.get())
                
                if midi_file:
                    wav_file = convert_midi_to_wav(midi_file, os.path.join(output_dir, f"{base_name}.wav"), config.SOUND_FONT_PATH)
                    if wav_file:
                        self.show_toast(f"Generirana melodija: {os.path.basename(wav_file)}", bootstyle=SUCCESS)
                        self.current_best_melody_wav_path = wav_file
                        
                        self.root.after(0, self.draw_waveform, wav_file)
                        self.play_last_melody()
                        
            self.set_ui_state_ready("Spreman.")
        except Exception as e:
            self.log_to_ui(f"Kritična greška tokom GA: {e}\n{traceback.format_exc()}")
            self.show_toast("Kritična greška tokom GA.", bootstyle=DANGER)
            self.set_ui_state_ready("Kritična greška tokom GA.")
            self.root.after(0, lambda: self.progress_meter.configure(amountused=0, bootstyle='danger'))

    # Pokreće reprodukciju posljednje generirane melodije
    def play_last_melody(self):
        if not self.current_best_melody_wav_path: return
        if pygame.mixer.music.get_busy():
            self.show_toast("Reprodukcija je već u toku.", bootstyle=INFO)
            return

        self.set_ui_state_busy("Reproduciram...")
        self.stop_button.config(state=NORMAL)
        try:
            pygame.mixer.music.load(self.current_best_melody_wav_path)
            pygame.mixer.music.play()
            self.start_animation() # Pokreće animaciju playhead-a
            self._check_playback_status() # Pokreće provjeru završetka reprodukcije
        except pygame.error as e:
            self.show_toast(f"Greška pri reprodukciji: {e}", bootstyle=DANGER)
            self.set_ui_state_ready("Greška.")

    # Periodično provjerava je li reprodukcija završila kako bi se ažurirao interfejs
    def _check_playback_status(self):
        if pygame.mixer.music.get_busy():
            self.playback_check_job = self.root.after(100, self._check_playback_status)
        else:
            self.stop_animation()
            self.stop_button.config(state=DISABLED)
            self.set_ui_state_ready("Reprodukcija je završena.")
            self.playback_check_job = None
    
    # Odmah zaustavlja audio reprodukciju i animaciju
    def stop_playback(self):
        self.stop_animation()
        
        pygame.mixer.music.stop()
        
        if self.playback_check_job:
            self.root.after_cancel(self.playback_check_job)
            self.playback_check_job = None

        if self.root.winfo_exists(): 
            self.stop_button.config(state=DISABLED)
        self.set_ui_state_ready("Spreman.")
        
    # Prikazuje poruke u konzoli 
    def log_to_ui(self, message):
        print(message)
        
    # Ažurira tekstualnu oznaku pored slider-a kako bi prikazala trenutnu vrijednost
    def _update_slider_label(self, value_str, label_widget, var_instance):
        try:
            if isinstance(var_instance, tk.DoubleVar):
                label_widget.config(text=f"{float(value_str)/100.0:.2f}")
            else:
                label_widget.config(text=f"{int(float(value_str))}")
        except (ValueError, tk.TclError):
            pass 
            
    # Ažurira tekst u statusnoj traci
    def update_status_bar(self, message):
        if self.root.winfo_exists():
            self.root.after(0, lambda: self.status_var.set(message))
            
    # Onemogućuje kontrole interfejsa dok je aplikacija zauzeta 
    def set_ui_state_busy(self, busy_message="Obrađujem..."):
        if self.root.winfo_exists():
            self.root.after(0, self._do_set_ui_state_busy, busy_message)
            
    # Interna metoda koja mijenja stanje dugmadi
    def _do_set_ui_state_busy(self, busy_message):
        for button in [self.run_ga_button, self.load_midi_button, self.play_button, self.browse_midi_button, self.export_style_button, self.import_style_button]:
            if button:
                button.config(state=DISABLED)
        self.status_var.set(busy_message)
        self.root.update_idletasks()
        
    # Omogućuje kontrole interfejsa kada je aplikacija spremna za novu akciju
    def set_ui_state_ready(self, ready_message="Spreman."):
        if self.root.winfo_exists():
            self.root.after(0, self._do_set_ui_state_ready, ready_message)
            
    # Interna metoda za određivanje koje dugmadi treba omogućiti
    def _do_set_ui_state_ready(self, ready_message):
        can_play_last = self.current_best_melody_wav_path and os.path.exists(self.current_best_melody_wav_path)
        is_playing = pygame.mixer.music.get_busy()
        
        self.run_ga_button.config(state=NORMAL if self.can_run_ga_flag and not is_playing else DISABLED)
        self.load_midi_button.config(state=NORMAL if not is_playing else DISABLED)
        self.browse_midi_button.config(state=NORMAL if not is_playing else DISABLED)
        self.import_style_button.config(state=NORMAL if not is_playing else DISABLED)
        self.export_style_button.config(state=NORMAL if self.can_run_ga_flag and self.style_evaluator and not is_playing else DISABLED)
        self.play_button.config(state=NORMAL if can_play_last and not is_playing else DISABLED)
        
        self.status_var.set(ready_message)
        self.root.update_idletasks()
        
    # Provjerava postojanje SoundFont datoteke potrebne za pretvorbu MIDI u WAV
    def check_soundfont(self):
        if not os.path.exists(config.SOUND_FONT_PATH):
            self.show_toast(f"UPOZORENJE: SoundFont '{config.SOUND_FONT_PATH}' nije pronađen.", bootstyle=WARNING, duration=6000)
            
    # Pokreće dugotrajne zadatke u pozadinskoj niti kako se korisnički interfejs ne bi zamrznuo
    def start_worker_thread(self, target_function, *args, **kwargs):
        if self.worker_thread and self.worker_thread.is_alive():
            messagebox.showwarning("Zauzeto", "Prethodna operacija je još uvijek u toku.")
            return
        self.stop_event.clear() 
        self.worker_thread = threading.Thread(target=target_function, args=args, kwargs=kwargs, daemon=True)
        self.worker_thread.start()
        
    # Otvara dijalog za odabir direktorija s MIDI datotekama i pokreće učenje stila
    def browse_midi_folder(self):
        folder_selected = filedialog.askdirectory(initialdir=self.midi_folder_path_var.get())
        if folder_selected:
            self.midi_folder_path_var.set(folder_selected)
            self.start_worker_thread(self.initialize_style_model, from_dataset=True)
            
    # Vraća putanju do datoteke zadanog stilskog modela
    def get_default_model_path(self):
        return os.path.join(config.BASE_DIR, config.DEFAULT_MODEL_FILENAME)
        
    # Pokušava učitati zadani stilski model ako postoji.
    def try_import_default_style_model(self):
        default_model_path = self.get_default_model_path()
        if os.path.exists(default_model_path):
            self.start_worker_thread(self.initialize_style_model, from_dataset=False, model_path=default_model_path)
            return True
        return False
        
    # Automatski sprema novonaučeni stil kao zadani model za buduću upotrebu
    def try_export_default_style_model(self):
        if self.style_evaluator and self.can_run_ga_flag:
            self.export_style_model(self.get_default_model_path())
            
    # Otvara dijalog za spremanje trenutno aktivnog stilskog modela u .pkl datoteku
    def export_style_model_dialog(self):
        if not self.style_evaluator:
            self.show_toast("Nema modela za izvoz.", bootstyle=WARNING)
            return
        file_path = filedialog.asksaveasfilename(defaultextension=".pkl", filetypes=[("Pickle files", "*.pkl")], initialdir=config.BASE_DIR, title="Izvezi Stilski Model")
        if file_path:
            self.export_style_model(file_path)
            
    # Sprema (serializira) statističke podatke stilskog modela koristeći pickle
    def export_style_model(self, file_path):
        try:
            with open(file_path, 'wb') as f:
                pickle.dump({
                    'pc_dist': self.style_evaluator.style_pitch_class_dist,
                    'int_dist': self.style_evaluator.style_interval_dist,
                    'bigr_dist': self.style_evaluator.style_pitch_class_bigram_dist,
                    'dur_dist': self.style_evaluator.style_duration_dist,
                    'ioi_dist': self.style_evaluator.style_ioi_dist,
                    'weights': self.style_evaluator.weights
                }, f)
            self.show_toast(f"Stilski model spremljen u {os.path.basename(file_path)}", bootstyle=SUCCESS)
        except Exception as e:
            self.show_toast(f"Greška pri spremanju modela: {e}", bootstyle=DANGER)
            
    # Otvara dijalog za učitavanje stilskog modela iz .pkl datoteke
    def import_style_model_dialog(self):
        file_path = filedialog.askopenfilename(defaultextension=".pkl", filetypes=[("Pickle files", "*.pkl")], initialdir=config.BASE_DIR, title="Uvezi Stilski Model")
        if file_path:
            self.start_worker_thread(self.initialize_style_model, from_dataset=False, model_path=file_path)
            
    # Pokreće genetski algoritam u pozadinskoj niti
    def start_ga_worker(self):
        if not self.can_run_ga_flag or not self.style_evaluator:
            self.show_toast("Stilski model nije spreman!", bootstyle=WARNING)
            return
        self.set_ui_state_busy("Pokrećem Genetski Algoritam...")
        self.start_worker_thread(self.run_ga_logic)
        
    # Otvara direktorij u kojem se spremaju generirane melodije.
    def open_output_folder(self):
        output_dir = os.path.join(config.BASE_DIR, config.OUTPUT_DIR_NAME)
        os.makedirs(output_dir, exist_ok=True)
        try:
            if os.name == 'nt': # Windows
                os.startfile(output_dir)
            elif sys.platform == 'darwin': # macOS
                subprocess.run(['open', output_dir])
            else: # Linux
                subprocess.run(['xdg-open', output_dir])
        except Exception as e:
            self.show_toast(f"Nije moguće otvoriti direktorij: {e}", bootstyle=WARNING)
            
    # Glavna funkcija za inicijalizaciju stilskog modela
    def initialize_style_model(self, from_dataset=True, model_path=None):
        if from_dataset:
            self.set_ui_state_busy("Učim stil iz skupa podataka...")
        elif model_path:
            self.set_ui_state_busy(f"Učitavam stilski model...")
            
        self.can_run_ga_flag = False
        current_style_evaluator = StyleEvaluator()
        success = False
        
        if from_dataset:
            dataset_path = self.midi_folder_path_var.get()
            if not os.path.isdir(dataset_path):
                self.show_toast(f"Putanja '{os.path.basename(dataset_path)}' nije ispravan direktorij.", bootstyle=DANGER)
                self.set_ui_state_ready("Greška: Neispravna putanja.")
                return
            success = current_style_evaluator.learn_style_from_dataset(dataset_path)
        elif model_path and os.path.exists(model_path):
            try:
                with open(model_path, 'rb') as f:
                    data_to_load = pickle.load(f)
                current_style_evaluator.style_pitch_class_dist = data_to_load.get('pc_dist')
                current_style_evaluator.style_interval_dist = data_to_load.get('int_dist')
                current_style_evaluator.style_pitch_class_bigram_dist = data_to_load.get('bigr_dist')
                current_style_evaluator.style_duration_dist = data_to_load.get('dur_dist')
                current_style_evaluator.style_ioi_dist = data_to_load.get('ioi_dist')
                current_style_evaluator.weights = data_to_load.get('weights', current_style_evaluator.weights)
                success = True
            except Exception as e:
                self.show_toast(f"Greška pri učitavanju modela: {e}", bootstyle=DANGER)
                
        # Ažuriranje stanja aplikacije ovisno o uspjehu
        if success:
            self.style_evaluator = current_style_evaluator
            self.can_run_ga_flag = True
            if from_dataset:
                self.try_export_default_style_model() 
            self.set_ui_state_ready("Spreman. Stilski model je aktivan.")
            self.show_toast("Stilski model uspješno učitan/naučen.", bootstyle=SUCCESS)
        else:
            self.style_evaluator = None
            self.set_ui_state_ready("Greška: Stilski model nije učitan/naučen.")
            self.show_toast("Neuspješno učitavanje stilskog modela.", bootstyle=DANGER)

