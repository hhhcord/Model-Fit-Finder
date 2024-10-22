import numpy as np
import os
import tkinter as tk  # Import the tkinter module (for GUI-related functions)
from tkinter import filedialog  # Import functions related to file selection dialog
import soundfile as sf  # Import the soundfile module (used for reading/writing audio files)
from scipy.io import wavfile  # Import the wavfile module

class AudioLoader:
    def __init__(self):
        # Specify the format of the audio files
        self.file_types = [('Audio Files', '*.wav *.mp3 *.m4a')]

    # Added a parameter to specify the number of seconds to load in the load_audio method
    def load_audio(self, seconds=5):
        root = tk.Tk()  # Create the root window for Tkinter
        root.withdraw()  # Hide the root window
        file_path = filedialog.askopenfilename(filetypes=self.file_types)  # Display file selection dialog and get the file path
        if file_path:
            print(f"User selected {file_path}")  # Display the file path selected by the user
            # First, open the file and get the sampling rate
            data, fs = sf.read(file_path, dtype='float32')
            # Re-read the data for the length of time specified by the user
            data, fs = sf.read(file_path, start=0, stop=seconds*fs, dtype='float32')
            return data, fs
        else:
            print("User selected Cancel")  # If the user cancels
            return None, None  # Return None
        
    def save_audio(self, time_series_data, sample_rate, file_name):
        # Normalize the time series data to ensure the range is between -1 and 1
        max_val = np.max(np.abs(time_series_data))
        if max_val > 0:
            time_series_data = time_series_data / max_val

        # Path for the output directory
        output_dir = './output'
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)  # Create the directory if it does not exist

        # Full path of the file to save
        output_path = os.path.join(output_dir, f"{file_name}.wav")

        # Save as a WAV file with float32 format
        wavfile.write(output_path, sample_rate, time_series_data.astype(np.float32))

        print(f"Audio saved to {output_path}")
