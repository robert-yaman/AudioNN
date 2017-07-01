import random
import os
import constants
import csv
import my_midi

TMP_DATA = '/tmp/data_tmp.csv'
TMP_LABELS = '/tmp/labels_tmp.csv'

def _nfile_len(f):
    return sum(1 for line in open(f))

def shuffle_data(training_data_csv=constants.TRAINING_DATA_PATH,
        training_labels_csv=constants.TRAINING_LABELS_PATH,
        validation_data_csv=constants.VALIDATION_DATA_PATH,
        validation_labels_csv=constants.VALIDATION_LABELS_PATH):
    pairs = [[training_data_csv, training_labels_csv], [validation_data_csv,
        validation_labels_csv]]
    for pair in pairs:
        data, labels = pair[0], pair[1]
        with open(data, 'rb') as data_file:
            with open(labels, 'rb') as labels_file:
                with open(TMP_DATA, 'ab') as tmp_data:
                    with open(TMP_LABELS, 'ab') as tmp_labels:
                        # Is there a non-terrible way? Could only load one file into
                        # memory at a time.
                        data_lines = data_file.readlines()
                        labels_lines = labels_file.readlines()

                        shuffled_indices = list(range(_file_len(data)))
                        random.shuffle(shuffled_indices)
                        for index in shuffled_indices:
                            tmp_data.write(data_lines[index])
                            tmp_labels.write(labels_lines[index])

                        os.system("rm %s" % data)
                        os.system("rm %s" % labels)
                        os.system("mv %s %s" %(TMP_DATA, data))
                        os.system("mv %s %s" %(TMP_LABELS, labels))


def write_shuffled_lines(lines, data_type, pct_validation=5):
    '''Takes a list of lines and randomly adds them to the main
    csvs. pct_validation percent of the data is written to the
    validation csvs.

    data_types is either 'features' or 'labels'
    '''
    if data_type == 'features':
        training_write_path = constants.TRAINING_DATA_PATH
        validation_write_path = constants.VALIDATION_DATA_PATH
    elif data_type == 'labels':
        training_write_path = constants.TRAINING_LABELS_PATH
        validation_write_path = constants.VALIDATION_LABELS_PATH
    else:
        print "ERROR: uncrecognized data type: %s" % data_type
        return

    rand_lines = lines[:]
    random.shuffle(rand_lines) 

    for line in rand_lines:
        if random.randrange(100 / pct_validation) == 1:
            with open(validation_write_path, 'a') as csv_file:
                writer = csv.writer(csv_file, delimiter=',')
                writer.writerow(line)
        else:
            with open(training_write_path, 'a') as csv_file:
                writer = csv.writer(csv_file, delimiter=',')
                writer.writerow(line)

if __name__ == "__main__":
    path = "training_data/raw/aria.mid"
    m = my_midi.MyMidi(path)
    write_shuffled_lines(m.labels(), 'labels')