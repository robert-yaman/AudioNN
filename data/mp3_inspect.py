import librosa
import librosa.display
import os
import numpy as np
import csv

def mfccFromTimeSeries(timeSeries, sr):
    return np.rot90(librosa.feature.mfcc(timeSeries, sr=sr))

def mfccFromPath(audio_path):
    # Returns np 2darray of shape (20, x). Each MFCC has length .023217s
    times_series, sr = librosa.load(audio_path)
    return mfccFromTimeSeries(times_series, sr)

def isAudio(filename):
    audio_extensions = ['wav','mp3','m4a']
    is_audio = False
    for extension in audio_extensions:
        if filename.endswith(extension):
            is_audio = True
    return is_audio

if __name__ == "__main__":
    # Take all of the audio files in training_labels/raw/, process them and add
    # them to training_labels/training_labels.csv.
    # TODO: Need to clip trailing silence so there is the same number of
    # examples in labels and data. Don't actually have to distinuish between
    # tracks as long as the index numbers are matched up correctly.
    raw_data_path = './training_labels/raw/'
    csv_path = './training_labels/training_labels.csv'

    with open(csv_path, 'wb') as csv_file:
        writer = csv.writer(csv_file, delimiter=',')
        for filename in os.listdir(raw_data_path):
            if isAudio(filename):
                print 'Processing ' + filename
                mfcc = mfccFromPath(raw_data_path + filename)
                writer.writerows(mfcc)
                # writer.writerow('testing')
    print 'Created csv at ' + csv_path 
