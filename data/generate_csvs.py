import csv
import os
import midi_inspect
import mp3_inspect
import random
import constants

def main():
    '''Script to generate data for training and validation.'''
    for midi_file in os.listdir(constants.AUDIO_FILE_PATH):
        if not midi_file.endswith(".mid"):
            continue
        # prints just filenames
        song_name = midi_file.split(".")[0]
        print "Processing " + song_name
        midi_path = constants.AUDIO_FILE_PATH + midi_file
        mp3_path = constants.MIDI_FILE_PATH +song_name + ".mp3"
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
            with open(constants.VALIDATION_DATA_PATH, "a") as csv_file:
                vd_writer = csv.writer(csv_file, delimiter=',')
                vd_writer.writerows(training)
            with open(constants.VALIDATION_LABELS_PATH, "a") as csv_file:
                vl_writer = csv.writer(csv_file, delimiter=',')
                vl_writer.writerows(labels)
        else:
            with open(constants.TRAINING_DATA_PATH, "a") as csv_file:
                td_writer = csv.writer(csv_file, delimiter=',')
                td_writer.writerows(training)
            with open(constants.TRAINING_LABELS_PATH, "a") as csv_file:
                tl_writer = csv.writer(csv_file, delimiter=',')
                tl_writer.writerows(labels)

if __name__ == '__main__':
    main()
