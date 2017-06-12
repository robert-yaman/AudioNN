import model
import tensorflow as tf

from tensorflow.python import debug as tf_debug

DATA_DIR = 'data/'
TRAINING_DATA = DATA_DIR + 'training_data/training_data.csv'
TRAINING_LABELS = DATA_DIR + 'training_labels/training_labels.csv'
VALIDATION_DATA = DATA_DIR + 'validation_data/validation_data.csv'
VALIDATION_LABELS = DATA_DIR + 'validation_labels/validation_labels.csv'

BATCH_SIZE = 50

def _get_data(validation=False):
    with tf.name_scope('get_data'):
        data_reader = tf.TextLineReader()
        feature_file = tf.train.string_input_producer([VALIDATIOIN_DATA if validation else
                TRAINING_DATA])
        _, csv_row = data_reader.read(feature_file)
        record_defaults = [[0.0]] * 20
        features = tf.stack(list(tf.decode_csv(csv_row,
            record_defaults=record_defaults)))

        label_reader = tf.TextLineReader()
        labels_file = tf.train.string_input_producer([VALIDATION_LABELS if validation else
                TRAINING_LABELS])
        _, csv_row = label_reader.read(labels_file)
        record_defaults = [[0]] * 88
        labels = tf.stack(list(tf.decode_csv(csv_row,
            record_defaults=record_defaults)))

        return features, labels

def input_pipeline(validation=False):
    example, label = _get_data(validation=validation)
    min_after_dequeue = 10000
    capacity = min_after_dequeue + 3 * BATCH_SIZE
    example_batch, label_batch = tf.train.shuffle_batch(
        [example, label], batch_size=BATCH_SIZE, capacity=capacity,
        min_after_dequeue=min_after_dequeue)
    label_batch = tf.cast(label_batch, tf.float32)
    return example_batch, label_batch

def main(argv=None):
    example_batch, label_batch = input_pipeline()
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
        coord = tf.train.Coordinator()
        # Start imperative step.
        threads = tf.train.start_queue_runners(coord=coord)
        try:
            step = 0
            while not coord.should_stop():
                step += 1
                print "Step: %d" % step
                if step % 1000 == 0:
                    loss_val, summary_val = sess.run([loss, summary], feed_dict={keep_prob:1.0})
                    print('Step: %d\n    Loss: %f' %(step, loss_val))
                    train_writer.add_summary(summary_val, step)
                else:
                    _, summary_val = sess.run([training_step, summary],
                            feed_dict={keep_prob:0.5})
                    test_writer.add_summary(summary_val, step)
        except tf.errors.OutOfRangeError:
            print("DONE TRAINING")
        finally:
            coord.request_stop()
            coord.join(threads)

if __name__ == '__main__':
   tf.app.run()
