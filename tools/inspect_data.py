from data import midi_inspect
from data import mp3_inspect
import librosa
import sys

midi = midi_inspect.labelsForPath("./data/training_data/raw/jg15_1.mid")
mp3 = mp3_inspect.stftFromPath("./data/training_labels/raw/jg15_1.mp3")
print "midi:"
print len(midi)
print "mp3:"
print len(mp3)
print "lenght:"
print librosa.get_duration(filename="./data/training_labels/raw/jg15_1.mp3")
print "trimmed length: "

time_series, _ = librosa.load("./data/training_labels/raw/jg15_1.mp3")
trimmed, _ = librosa.effects.trim(time_series)
print librosa.get_duration(trimmed[1540:])
print "trimmed interval"
print _
