import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np
from librosa.feature.rhythm import tempo  # for librosa >= 0.10, v0.11.0 was released on 2025-03-11
import os

# Load audio file from my folder or mp3s
filename = '/Users/jeffersonsuh/Library/Mobile Documents/com~apple~CloudDocs/grand25_SOUNDS/0_sample the _ out of these/NJZ - Get Up.mp3'
y, sr = librosa.load(filename, duration=30.0)

# Get just the file name (without the path) for our visual
track_name = os.path.basename(filename)

# Estimate multiple tempo candidates
tempos = tempo(y=y, sr=sr, aggregate=None) # y is audio waveform (1D NumPy array), sr is the sampling rate, agg so vector of tempos and not single int
top_tempo = tempos[0]
print("Tempo candidates (BPM):", np.round(tempos, 2)) # round to 120.00
print(f"Top Estimated Tempo: {top_tempo:.2f} BPM") # format the float with 2 digits after the decimal point

# Estimate beat frames using beat tracker
beat_tempo, beat_frames = librosa.beat.beat_track(y=y, sr=sr) # the magic! bpm as a float, array of ints that each are a frame index in the STFT window

# Convert beat frames to time
beat_times = librosa.frames_to_time(beat_frames, sr=sr) # frames * hop_length / sr = tiem value in seconds.

# Plot waveform and beat markers
plt.figure(figsize=(14, 5))
librosa.display.waveshow(y, sr=sr, alpha=0.6)
plt.vlines(beat_times, -1, 1, color='r', linestyle='--', label='Beats')
plt.title(f"{track_name} — Tempo: {top_tempo:.2f} BPM")
plt.legend()
plt.tight_layout()
plt.show()
