import model
import tensorflow as tf

from tensorflow.python import debug as tf_debug

BATCH_SIZE = 50

def _data_dir(local):
    return 'data/' if local else 'gs://audionn-data/'

def _training_data_path(local):
    file_name = 'training_data/training_data.csv' if local else 'training_data.csv'
    return _data_dir(local) + file_name

def _training_labels_path(local):
    file_name = 'training_labels/training_labels.csv' if local else 'training_labels.csv'
    return _data_dir(local) + file_name

def _validation_data_path(local):
    file_name = 'validation_data/validation_data.csv' if local else 'validation_data.csv'
    return _data_dir(local) + file_name

def _validation_labels_path(local):
    file_name = 'validation_labels/validation_labels.csv' if local else 'validation_labels.csv'
    return _data_dir(local) + file_name

def _get_file_len(path):
    """
    with open(path, 'rb') as f:
        for i,l in enumerate(f):
            pass
    return i
    """
    # Can't read file len since stored in GS - find another way
    return 2800000

def _get_data(local, validation=False):
    with tf.name_scope('get_data'):
        data_reader = tf.TextLineReader()
        if validation:
            feature_file = tf.train.string_input_producer([_validation_data_path(local)])
        else:
            feature_file = tf.train.string_input_producer([_training_data_path(local)])
        _, csv_row = data_reader.read(feature_file)
        record_defaults = [[0.0]] * 20
        features = tf.stack(list(tf.decode_csv(csv_row,
            record_defaults=record_defaults)))

        label_reader = tf.TextLineReader()
        if validation:
            labels_file = tf.train.string_input_producer([_validation_labels_path(local)])
        else:
            labels_file = tf.train.string_input_producer([_training_labels_path(local)])
        _, csv_row = label_reader.read(labels_file)
        record_defaults = [[0]] * 88
        labels = tf.stack(list(tf.decode_csv(csv_row,
            record_defaults=record_defaults)))

        return features, labels

def input_pipeline(local, validation=False):
    example, label = _get_data(local, validation=validation)
    min_after_dequeue = 10000
    capacity = min_after_dequeue + 3 * BATCH_SIZE
    example_batch, label_batch = tf.train.shuffle_batch(
        [example, label], batch_size=BATCH_SIZE, capacity=capacity,
        min_after_dequeue=min_after_dequeue)
    label_batch = tf.cast(label_batch, tf.float32)
    return example_batch, label_batch

def main(argv=None):
    local = "--local" in argv
    example_batch, label_batch = input_pipeline(local)
    readout, keep_prob = model.get_transcription_model(example_batch)

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
            # For each note, did it get the right prediction?
                # Should be >90% if it just predicts all 0s.
            correct_predictions7 = tf.cast(tf.equal(interpretation(readout,
                0.7), label_batch), tf.float32)
            correct_predictions5 = tf.cast(tf.equal(interpretation(readout,
                0.5), label_batch), tf.float32)
            correct_predictions9 = tf.cast(tf.equal(interpretation(readout,
                0.9), label_batch), tf.float32)

            # Did it get the right prediction for every note? Look if the
            # min value is 1.
            correct_prediction7 = tf.equal(tf.argmin(correct_predictions7,
                1),1)
            correct_prediction5 = tf.equal(tf.argmin(correct_predictions5,
                1),1)
            correct_prediction9 = tf.equal(tf.argmin(correct_predictions9,
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
        train_writer = tf.summary.FileWriter('/tmp/tensorboard/train', sess.graph)
        test_writer = tf.summary.FileWriter('/tmp/tensorboard/test')

        if "debug" in argv:
            sess = tf_debug.LocalCLIDebugWrapperSession(sess)
            sess.add_tensor_filter("has_inf_or_nan", tf_debug.has_inf_or_nan)

        sess.run(tf.global_variables_initializer())
        sess.run(tf.local_variables_initializer())
        coord = tf.train.Coordinator()
        # Start imperative steps.
        threads = tf.train.start_queue_runners(coord=coord)
        num_epochs = _get_file_len(_training_data_path(local)) / BATCH_SIZE
        step = 0
        while step < num_epochs and not coord.should_stop():
            step += 1
            print "Step: %d" % step
            if step % 1000 == 0:
                loss_val, summary_val = sess.run([loss, summary], feed_dict={keep_prob:1.0})
                print('Step: %d\n    Loss: %f' %(step, loss_val))
                train_writer.add_summary(summary_val, step)
            _, summary_val = sess.run([training_step, summary],
                    feed_dict={keep_prob:0.5})
            test_writer.add_summary(summary_val, step)
        print("DONE TRAINING")
        coord.request_stop()
        coord.join(threads)

if __name__ == '__main__':
   tf.app.run()
