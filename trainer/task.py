'''
Possible flags:
    - "--debug": runs in tf debugger.
    - "--local" signifies training is running on local CPU.
'''

import model
import tensorflow as tf
from tensorflow.python import debug as tf_debug
import sys
import os

sys.path.insert(1, os.path.join(sys.path[0], '..'))
from data import constants


BATCH_SIZE = 50

def _data_dir(local):
    return constants.VOLUME_PATH if local else 'gs://audionn-data/'

def _training_data_path(local):
    return _data_dir(local) + 'training_data.csv'

def _training_labels_path(local):
    return _data_dir(local) + 'training_labels.csv'

def _validation_data_path(local):
    return _data_dir(local) + 'validation_data.csv'

def _validation_labels_path(local):
    return _data_dir(local) + 'validation_labels.csv'

def _get_training_length(path):
    # Can't read file len since stored in GS - find another way
    return 2800000

def _get_data(validation, local):
    with tf.name_scope('get_data'):
        data_reader = tf.TextLineReader()
        data_path_fn = _validation_data_path if validation else _training_data_path
        feature_file = tf.train.string_input_producer([data_path_fn(local)])
        _, csv_row = data_reader.read(feature_file)
        record_defaults = [[0.0]] * constants.AUDIO_FEATURE_COUNT
        features = tf.stack(list(tf.decode_csv(csv_row,
            record_defaults=record_defaults)))

        label_reader = tf.TextLineReader()
        label_path_fn = _validation_labels_path if validation else _training_labels_path
        labels_file = tf.train.string_input_producer([label_path_fn(local)])
        _, csv_row = label_reader.read(labels_file)
        record_defaults = [[0]] * 88
        labels = tf.stack(list(tf.decode_csv(csv_row,
            record_defaults=record_defaults)))

        return features, labels


def input_pipeline(validation, local, batch_size=BATCH_SIZE):
    example_line, label_line = _get_data(validation, local)
    min_after_dequeue = 10000
    capacity = min_after_dequeue + 3 * batch_size
    example_batch, label_batch = tf.train.shuffle_batch(
        [example_line, label_line], batch_size=batch_size, capacity=capacity,
        min_after_dequeue=min_after_dequeue)
    label_batch = tf.cast(label_batch, tf.float32)
    return example_batch, label_batch

def main(argv=None):
    local = "--local" in argv

    example_batch, label_batch = input_pipeline(False, local)
    # 5% validation data. Figure out a better way to halt (or re-use?). Try num_epochs=None.
    v_example, v_label = input_pipeline(True, local, batch_size=BATCH_SIZE * 20)

    with tf.variable_scope("model") as scope:
        readout, keep_prob = model.get_transcription_model(example_batch)
        # Use the same weights and biases for the validation model.
        scope.reuse_variables()
        v_readout, v_keep_prob = model.get_transcription_model(v_example)

    # Don't use softmax since the outputs aren't mutually exclusive.
    with tf.name_scope('train'):
        loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=label_batch,
                logits=readout))
        tf.summary.scalar('loss', loss)
        training_step = tf.train.AdamOptimizer(1e-4).minimize(loss)

    def interpretation(logit, cutoff):
        # Returns a tensor that represents how we are interpreting the output
        # of our model.
        # |logit| is a [1, 88] tensor representing the likelihood that each
        # note is present in a given sample.
        # |cutoff| is the probability at which we will consider a note to be
        # present.
        with tf.name_scope('interpretation'):
            return tf.cast(tf.greater(logit, cutoff), tf.float32)

    # For now, analyze successfulness as number of predictions it gets
    # exactly right. We can revisit this later.
    with tf.name_scope('accuracy'):
        with tf.name_scope('predictions'):
            sigmoid = tf.sigmoid(v_readout)
            # For each note, did it get the right prediction?
                # Should be >90% if it just predicts all 0s.
            correct_predictions7 = tf.cast(tf.equal(interpretation(sigmoid,
                0.7), v_label), tf.float32)
            correct_predictions5 = tf.cast(tf.equal(interpretation(sigmoid,
                0.5), v_label), tf.float32)
            correct_predictions9 = tf.cast(tf.equal(interpretation(sigmoid,
                0.9), v_label), tf.float32)

            # Did it get the right prediction for every note? Look if the
            # min value is 1.
            correct_prediction7 = tf.equal(tf.reduce_min(correct_predictions7,
                1),1)
            correct_prediction5 = tf.equal(tf.reduce_min(correct_predictions5,
                1),1)
            correct_prediction9 = tf.equal(tf.reduce_min(correct_predictions9,
                1),1)

        accuracy7 = tf.reduce_mean(tf.cast(correct_prediction7,
            tf.float32))
        accuracy5 = tf.reduce_mean(tf.cast(correct_prediction5,
            tf.float32))
        accuracy9 = tf.reduce_mean(tf.cast(correct_prediction9,
            tf.float32))

        tf.summary.scalar('accuracy7', accuracy7)
        tf.summary.scalar('accuracy5', accuracy5)
        tf.summary.scalar('accuracy9', accuracy9)

    summary = tf.summary.merge_all()
    with tf.Session() as sess:
        tb_path = '/tmp/tensorboard/' if local else 'gs://audionn-data/tensorboard/'
        summary_writer = tf.summary.FileWriter(tb_path, sess.graph)

        if "--debug" in argv:
            sess = tf_debug.LocalCLIDebugWrapperSession(sess)
            sess.add_tensor_filter("has_inf_or_nan", tf_debug.has_inf_or_nan)

        sess.run(tf.global_variables_initializer())
        sess.run(tf.local_variables_initializer())
        coord = tf.train.Coordinator()
    
        # Start imperative steps.
        threads = tf.train.start_queue_runners(coord=coord)
        num_steps = _get_training_length(_training_data_path(local)) / BATCH_SIZE

        print "BEGINNING TRANING"
        step = 0
        while step < num_steps and not coord.should_stop():
            step += 1
            print step
            sess.run([training_step], feed_dict={keep_prob:0.5})
            if step % 1000 == 0:
                # Do I already get accuracies from the summary?
                l, s, a5, a7, a9 = sess.run([loss, summary, accuracy5, accuracy7, accuracy9], 
                    feed_dict={v_keep_prob:1.0})
                print('Step: %d    Loss: %f\n    Accuracies: %d, %d, %d' % 
                    (step, l, a5, a7, a9))
                summary_writer.add_summary(s, step)
        print("DONE TRAINING")
        coord.request_stop()
        coord.join(threads)

if __name__ == '__main__':
    tf.app.run()
