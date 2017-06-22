import random
import os

TRAINING_DATA_CSV = 'training_data/training_data.csv'
TRAINING_LABELS_CSV = 'training_labels/training_labels.csv'
VALIDATION_DATA_CSV = 'validation_data/validation_data.csv'
VALIDATION_LABELS_CSV = 'validation_labels/validation_labels.csv'

TMP_DATA = '/tmp/data_tmp.csv'
TMP_LABELS = '/tmp/labels_tmp.csv'

def file_len(f):
    with open(f, 'rb'):
        for i,l in enumerate(f):
            pass
        return i

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
                        #data_reader = csv.reader(data_file)
                        #labels_reader = csv.reader(labels_file)
                        # Is this terrible? Could only load one file into
                        # memory at a time.
                        data_lines = data_file.readlines()
                        labels_lines = labels_file.readlines()
                        #data_writer = csv.writer(tmp_data)
                        #labels_writer = csv.writer(tmp_labels)

                        shuffled_indices = random.shuffle(list(range(file_len(data_file))))
                        for index in shuffled_indicies:
                            tmp_data.write(data_lines[index])
                            tmp_labels.write(labels_lines[index])

                        os.system("rm %s" % data)
                        os.system("rm %s" % labels)
                        os.system("mv %s %s" % TMP_DATA, data)
                        os.system("mv %s %s" % TMP_LABELS, labels)


if __name__ == "__main__":
    shuffle_data(training_data_csv="td.csv",
            training_labels_csv="tl.csv",
            validation_data_csv="vd.csv",
            validation_labels_csv="vl.csv")
