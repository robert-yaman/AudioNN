'''Script to generate data for training and validation.
-r generates csvs for recurrent NN (non-randomized and separated by song).
'''

import csv
import os
import midi_inspect
import mp3_inspect
import random
import constants
import shuffle_data
import sys

def main():
    for midi_file in os.listdir(constants.MIDI_FILE_PATH):
        if not midi_file.endswith(".mid"):
            continue
        song_name = midi_file.split(".")[0]
        print "Processing " + song_name
        midi_path = constants.MIDI_FILE_PATH + midi_file
        mp3_file = song_name + ".mp3"
        mp3_path = constants.AUDIO_FILE_PATH + mp3_file

        try:
             labels = midi_inspect.labelsForPath(midi_path)
        except:
            print "<<<<<<<<< ERROR in MIDI: " + song_name
            continue
        try:
            training = mp3_inspect.cqtFromPath(mp3_path)
        except:
            print "<<<<<<< ERROR in MP3: " + song_name
            continue

        # Make sure there are the same amount of examples. Usually this means
        # trimming off the silence at the end of the audio.
        print "Original:"
        print "  %d" % len(training)
        print "  %d" % len(labels)
        if len(training) > len(labels):
            training = training[:len(labels)]
        elif len(labels) > len(training):
            labels = labels[:len(training)]
        print "Final:"
        print "  %d" % len(training)
        print "  %d" % len(labels)

        if '-r' in sys.argv:
            # Store per-file csvs. We might use this when we add a recurrent layer.
            training_csv_path = constants.TRAINING_DATA_DIRECTORY + song_name + '.csv'
            labels_csv_path = constants.TRAINING_LABELS_DIRECTORY + song_name + '.csv'

            with open(training_csv_path, 'a') as training_csv:
                training_writer = csv.writer(training_csv, delimiter=',')
                training_writer.writerows(training)

            with open(labels_csv_path, 'a') as labels_csv:
                labels_writer = csv.writer(labels_csv, delimiter=',')
                labels_writer.writerows(labels)
        else:
            shuffle_data.write_shuffled_lines(training, 'features')
            shuffle_data.write_shuffled_lines(labels, 'labels')




if __name__ == '__main__':
    main()
