import librosa
import librosa.display
import os
import numpy as np
import csv

def mfccFromTimeSeries(timeSeries, sr):
    return np.rot90(librosa.feature.mfcc(timeSersies, sr=sr))

def stftFromTimeSeries(timeSeries):
    return np.rot90(librosa.core.stft(timeSeries))

def mfccFromPath(audio_path):
    # Returns np 2darray of shape (20, x).
    times_series, sr = librosa.load(audio_path)
    return mfccFromTimeSeries(times_series, sr)

def stftFromPath(audio_path):
    # Returns np 2darray of shape (1024, x).
    time_series, _ = librosa.load(audio_path)
    trimmed, _ = librosa.effects.trim(time_series)
    # Timmidity inserts .07s of silence at the beginning of the track that is
    # hard to pick up with librosa.
    return stftFromTimeSeries(trimmed[1540:])

def _is_audio(filename):
    audio_extensions = ['wav','mp3','m4a']
    is_audio = False
    for extension in audio_extensions:
        if filename.endswith(extension):
            is_audio = True
    return is_audio