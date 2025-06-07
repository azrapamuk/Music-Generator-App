# main.py
import tkinter as tk
import os
import config

# Uvozimo klasu koja sadrži sav UI kod
from ui import MusicGeneratorApp

def main():
    """Glavna funkcija za pokretanje aplikacije."""
    
    # Provera i kreiranje potrebnih foldera
    if not os.path.exists(config.DRIVE_MIDI_FOLDER_PATH):
        try:
            os.makedirs(config.DRIVE_MIDI_FOLDER_PATH, exist_ok=True)
            print(f"Kreiran folder za MIDI podatke: {config.DRIVE_MIDI_FOLDER_PATH}")
        except OSError as e:
            print(f"Nije moguće kreirati folder {config.DRIVE_MIDI_FOLDER_PATH}: {e}")
    
    # Provera postojanja SoundFont-a
    if not os.path.exists(config.SOUND_FONT_PATH): 
        print(f"\n!!! UPOZORENJE: SoundFont fajl '{config.SOUND_FONT_FILENAME}' nije pronađen. !!!")
        print("Molimo preuzmite SoundFont i postavite ga u folder sa skriptom.")
    
    # Kreiranje glavnog prozora i instance aplikacije
    root = tk.Tk()
    app = MusicGeneratorApp(root)
    
    # Pokretanje glavne petlje (prikazuje prozor i čeka na akcije korisnika)
    root.mainloop()

if __name__ == "__main__":
    main()