import csv
import os
import midi_inspect
import mp3_inspect
import random 

TRAINING_DATA_CSV = 'training_data/training_data.csv'
TRAINING_DATA_PATH = 'training_data/raw/'
TRAINING_LABELS_CSV = 'training_labels/training_labels.csv'
TRAINING_LABELS_PATH = 'training_labels/raw/'
VALIDATION_DATA_CSV = 'validation_data/validation_data.csv'
VALIDATION_LABELS_CSV = 'validation_labels/validation_labels.csv'

def main():
    '''Script to generate data for training and validation.'''
    for midi_file in os.listdir(TRAINING_DATA_PATH):
        if not midi_file.endswith(".mid"):
            continue
        # prints just filenames
        song_name = midi_file.split(".")[0]
        print "Processing " + song_name
        midi_path = TRAINING_DATA_PATH + midi_file
        mp3_path = TRAINING_LABELS_PATH +song_name + ".mp3"
        # Do try statemtn
        try:
             labels = midi_inspect.labelsForPath(midi_path)
        except:
            print "<<<<<<<<< ERROR in MIDI: " + song_name
            continue
        try:
            training = mp3_inspect.mfccFromPath(mp3_path)
        except:
            print "<<<<<<< ERROR in MFCCs: " + song_name
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
        if random.randrange(20) == 1:
            with open(VALIDATION_DATA_CSV, "a") as csv_file:
                vd_writer = csv.writer(csv_file, delimiter=',')
                vd_writer.writerows(training)
            with open(VALIDATION_LABELS_CSV, "a") as csv_file:
                vl_writer = csv.writer(csv_file, delimiter=',')
                vl_writer.writerows(labels)
        else:
            with open(TRAINING_DATA_CSV, "a") as csv_file:
                td_writer = csv.writer(csv_file, delimiter=',')
                td_writer.writerows(training)
            with open(TRAINING_LABELS_CSV, "a") as csv_file:
                tl_writer = csv.writer(csv_file, delimiter=',')
                tl_writer.writerows(labels)

if __name__ == '__main__':
    main()
