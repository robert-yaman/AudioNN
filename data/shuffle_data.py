import random
import os

TRAINING_DATA_CSV = 'training_data/training_data.csv'
TRAINING_LABELS_CSV = 'training_labels/training_labels.csv'
VALIDATION_DATA_CSV = 'validation_data/validation_data.csv'
VALIDATION_LABELS_CSV = 'validation_labels/validation_labels.csv'

TMP_DATA = '/tmp/data_tmp.csv'
TMP_LABELS = '/tmp/labels_tmp.csv'

def file_len(f):
    return sum(1 for line in open(f))

def shuffle_data(training_data_csv=TRAINING_DATA_CSV,
        training_labels_csv=TRAINING_LABELS_CSV,
        validation_data_csv=VALIDATION_DATA_CSV,
        validation_labels_csv=VALIDATION_LABELS_CSV):
    '''Use Fisher Yates shuffle to create new tmp files, then replace the old
    files with these ones
    '''
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

                        shuffled_indices = list(range(file_len(data)))
                        random.shuffle(shuffled_indices)
                        for index in shuffled_indices:
                            tmp_data.write(data_lines[index])
                            tmp_labels.write(labels_lines[index])

                        os.system("rm %s" % data)
                        os.system("rm %s" % labels)
                        os.system("mv %s %s" %(TMP_DATA, data))
                        os.system("mv %s %s" %(TMP_LABELS, labels))


if __name__ == "__main__":
    shuffle_data()
