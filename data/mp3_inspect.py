import librosa
import librosa.display
import os
import numpy as np
import csv
import constants

def mfccFromTimeSeries(timeSeries, sr):
    return np.rot90(librosa.feature.mfcc(timeSeries, sr=sr))

def stftFromTimeSeries(timeSeries):
    return np.rot90(librosa.core.stft(timeSeries))

def cqtFromTimeSeries(timeSeries, sr):
    return np.abs(np.rot90(librosa.core.cqt(timeSeries, sr, 
        bins_per_octave=36, n_bins=constants.NUM_CQT_BINS)))

def mfccFromPath(audio_path):
    # Mel frequency cepstrum coefficients. Returns np 2darray of shape (20, x).
    times_series, sr = librosa.load(audio_path)
    return mfccFromTimeSeries(times_series, sr)

def stftFromPath(audio_path):
    # Short-term Fourier transform. Returns np 2darray of shape (1025, x).
    time_series, _ = librosa.load(audio_path)
    trimmed, _ = librosa.effects.trim(time_series)
    # timidity inserts .07s of silence at the beginning of the track that is
    # hard to pick up with librosa, so we manually trim it here.
    return stftFromTimeSeries(trimmed[1540:])

def cqtFromPath(audio_path):
    # Constant-Q Transform. Returns np 2darray of shape (NUM_CQT_BINS, x).
    time_series, sr = librosa.load(audio_path)
    trimmed, _ = librosa.effects.trim(time_series)
    # timidity inserts .07s of silence at the beginning of the track that is
    # hard to pick up with librosa, so we manually trim it here.
    return cqtFromTimeSeries(trimmed[1540:], sr)

def magnitudesFromPath(audio_path):
    stft = stftFromPath(audio_path)
    # We drop information about the cycle.
    mangitude_func = np.vectorize(lambda t: np.abs(t))
    return mangitude_func(stft)

def _is_audio(filename):
    audio_extensions = ['wav','mp3','m4a']
    is_audio = False
    for extension in audio_extensions:
        if filename.endswith(extension):
            is_audio = True
    return is_audio

if __name__ == "__main__":
    path = "./training_labels/raw/aria.mp3"
    # time_series, sr = librosa.load(path)
    # librosa.output.write_wav("/tmp/tmpwav.wav", time_series, sr, norm=False)
    cqt = cqtFromPath(path)
    print cqt
    print cqt.shape
