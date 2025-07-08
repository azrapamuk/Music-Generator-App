
# ♫⋆｡ Music Generator with Genetic Algorithm

This application uses a genetic algorithm to generate new musical melodies. It first learns a musical style from a given set of MIDI files, then evolves new melodies that stylistically resemble the learned pattern.

## Repository Structure

This GitHub repository contains the following folders:

- **`Music-Generator-App/`**  
  A desktop application with a graphical interface that allows users to generate and play melodies. All core features described below (style learning, melody generation, playback, etc.) are part of this app.

- **`Drive-Files/`**  
  A folder containing all supplementary files and resources that were part of the original project. These files are also available in the original project folder at: [Google Drive of the Project](https://drive.google.com/drive/folders/13-jPpJeRsy0dtSc-tFalxX1jc6NVA1eo)

## Main Features

- **Style Learning**: Analyzes a directory of MIDI files to learn statistical features of the musical style (pitch distribution, intervals, note durations, etc.).
- **Model Saving and Loading**: The learned style model can be saved as a `.pkl` file for later use.
- **Melody Generation**: Uses a genetic algorithm to generate new melodies based on the active style model.
- **Customizable Parameters**: Allows the user to adjust key parameters of the genetic algorithm (population size, number of generations, mutation and crossover rates).
- **Audio Visualization**: Displays the waveform of the generated melody.
- **Audio Playback**: Integrated player to listen to the generated `.wav` file.
- **Instrument Selection**: Option to choose a General MIDI instrument for the generated melody.

## Prerequisites

Before running, make sure the following are installed:
- **Python** (recommended version 3.8 or newer)
- **pip** (Python package installer, usually comes with Python)
- **FluidR3_GM Soundfont** (needs to be downloaded from [FluidR3_GM Soundfont](https://member.keymusician.com/Member/FluidR3_GM/index.html))

## Installation and Running Instructions

Follow these steps to set up and run the application:

1. Clone the repository  
2. Install all required packages using the `requirements.txt` file:  
   ```bash
   pip install -r requirements.txt
   ```  
3. Place the `FluidR3_GM.sf2` file you previously downloaded into the project directory  
4. Run the main script:  
   ```bash
   python main.py
   ```

## How to Use the Application

- **Loading the Style Model**:
  - *Option A (Learn from Data)*: In the "Path to MIDI Dataset" field, enter the path to the directory with MIDI files and click "Load Style from Dataset". Wait for the process to complete.
  - *Option B (Import Model)*: Click "Import Model (.pkl)" and select a previously saved `.pkl` file.

- **Adjusting GA Parameters**:
  - Use the sliders in section "2. GA Parameters" to customize the genetic algorithm settings as desired.

- **Selecting an Instrument**:
  - In the "3. Instrument" dropdown menu, choose your preferred instrument.

- **Generating a Melody**:
  - Click the "Generate Melody" button.

- **Playback and Management**:
  - Once the melody is generated, the waveform will be displayed and playback will start automatically.
  - Use the "Play" and "Stop" buttons to control playback.
  - Click "Open Directory" to view the saved `.mid` and `.wav` files.