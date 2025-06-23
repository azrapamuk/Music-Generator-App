# ♫⋆｡ Muzički Generator s Genetskim Algoritmom

Ova aplikacija koristi genetski algoritam za generiranje novih muzičkih melodija. Aplikacija prvo uči muzički stil iz zadanog skupa MIDI datoteka, a zatim evoluira nove melodije koje su stilski slične naučenom uzorku.

## Glavne Značajke

- **Učenje stila**: Analizira direktorij s MIDI datotekama kako bi naučila statističke značajke glazbenog stila (distribuciju visina tonova, intervala, trajanja nota, itd.).
- **Spremanje i učitavanje modela**: Naučeni stilski model može se spremiti kao `.pkl` datoteka za kasniju upotrebu.
- **Generiranje melodija**: Koristi genetski algoritam za stvaranje novih melodija na temelju aktivnog stilskog modela.
- **Prilagodljivi parametri**: Omogućuje korisniku da podešava ključne parametre genetskog algoritma (veličina populacije, broj generacija, stopa mutacije i križanja).
- **Audio vizualizacija**: Prikazuje zvučni val generirane melodije.
- **Reprodukcija zvuka**: Integrirani player za preslušavanje generiranog `.wav` zapisa.
- **Odabir instrumenta**: Mogućnost odabira General MIDI instrumenta za generiranu melodiju.

## Preduvjeti

Prije pokretanja, osigurajte da imate instalirano sljedeće:
- **Python** (preporučena verzija 3.8 ili novija)
- **pip** (alat za instalaciju Python paketa, obično dolazi s Pythonom)
- **FluidR3_GM Soundfont** (koji je potrebno preuzeti sa [FluidR3_GM Soundfont](https://member.keymusician.com/Member/FluidR3_GM/index.html))

## Upute za Instalaciju i Pokretanje

Slijedite ove korake kako biste postavili i pokrenuli aplikaciju.

1.  Klonirajte repozitorij
2. Instalirajte sve potrebne pakte pomoću requirements.txt datoteke
`pip install -r requirements.txt`
3. U direktorij projekta ubacite `FluidR3_GM.sf2` datoteku koju ste prethodno preuzeli
4. Pokrenite glavnu skriptu
`python main.py`

Kako Koristiti Aplikaciju
- Učitavanje stilskog modela:
  - Opcija A (Učenje iz podataka): U polje "Putanja do MIDI skupa podataka" unesite putanju do direktorija s MIDI datotekama i kliknite "Učitaj stil iz skupa podataka". Pričekajte da se proces završi.
  - Opcija B (Uvoz modela): Kliknite "Uvezi Model (.pkl)" i odaberite prethodno spremljenu .pkl datoteku.
- Podešavanje GA Parametara:
  - Koristite klizače u odjeljku "2. GA Parametri" kako biste prilagodili postavke genetskog algoritma prema vašim željama.
- Odabir Instrumenta:
  - U padajućem izborniku "3. Instrument" odaberite željeni instrument.
- Generiranje Melodije:
  - Kliknite dugme "Generiraj Melodiju". 
- Preslušavanje i Upravljanje:
  - Nakon generiranja, zvučni val će se prikazati, a melodija će automatski zasvirati.
- Koristite gumbe "Sviraj" i "Zaustavi" za kontrolu reprodukcije.
  - Kliknite "Otvori Direktorij" kako biste vidjeli spremljene .mid i .wav datoteke.
