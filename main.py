# main.py

import tkinter as tk
import os
import config

# Uvozimo klasu koja sadrži sav UI kod
from ui import MusicGeneratorApp

# Glavna funkcija koja inicijalizuje aplikaciju i pokreće grafički interfejs
def main():
    
    if not os.path.exists(config.DRIVE_MIDI_FOLDER_PATH):
        try:
            os.makedirs(config.DRIVE_MIDI_FOLDER_PATH, exist_ok=True)
            print(f"Kreiran folder za MIDI podatke: {config.DRIVE_MIDI_FOLDER_PATH}")
        except OSError as e:
            print(f"Nije moguće kreirati folder {config.DRIVE_MIDI_FOLDER_PATH}: {e}")
    
    if not os.path.exists(config.SOUND_FONT_PATH): 
        print(f"\n!!! UPOZORENJE: SoundFont fajl '{config.SOUND_FONT_FILENAME}' nije pronađen. !!!")
        print("Molimo preuzmite SoundFont i postavite ga u folder sa skriptom.")
    
    root = tk.Tk()
    app = MusicGeneratorApp(root)
    
    root.mainloop()

if __name__ == "__main__":
    main()
