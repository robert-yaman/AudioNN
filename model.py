import tensorflow as tf
import numpy

from tensorflow.python import debug as tf_debug

DATA_DIR = 'small_dataset/'
# TRAINING_STEPS * BATCH_SIZE has to be less than the amount of training data
# we have.
TRAINING_STEPS = 100
BATCH_SIZE = 50


# TODO: make number of MFCC units a parameter, since we will likely change

def variable_summaries(var):
    """Attach a lot of summaries to a Tensor (for TensorBoard
    visualization)."""
    with tf.name_scope('summaries'):
        mean = tf.reduce_mean(var)
        tf.summary.scalar('mean', mean)
        with tf.name_scope('stddev'):
            stddev = tf.sqrt(tf.reduce_mean(tf.square(var- mean)))
        tf.summary.scalar('stddev', stddev)
        tf.summary.scalar('max', tf.reduce_max(var))
        tf.summary.scalar('min',tf.reduce_min(var))
        tf.summary.histogram('histogram',var)

def main(argv=None):
    # Import data
    training_labels = numpy.load(DATA_DIR + 'training_labels.npy')
    # Data will have more data than labels, since data takes into account
    # trailing silence in recordings. We trim this.
    training_data = numpy.load(DATA_DIR +
            'training_data.npy')[0:len(training_labels)]
    validation_labels = numpy.load(DATA_DIR + 'validation_labels.npy')
    validation_data = numpy.load(DATA_DIR +
            'validation_data.npy')[0:len(validation_labels)]

    def weight_variables(shape):
        with tf.name_scope('weights'):
            initial = tf.truncated_normal(shape, stddev=.01)
            variables = tf.Variable(initial)
            variable_summaries(variables)
            return variables
    def bias_variables(shape):
        with tf.name_scope('biases'):
            initial = tf.constant(0.1, shape=shape)
            variables = tf.Variable(initial)
            variable_summaries(variables)
            return variables
    def conv_layer(inputs, filter_shape):
        with tf.name_scope('conv'):
            return tf.nn.conv1d(inputs, filter_shape, stride=1,
                    padding='SAME')
    def pooling_layer(inputs):
        with tf.name_scope('pool'):
            return tf.nn.pool(inputs, window_shape=[2], strides=[2], pooling_type='MAX',
                    padding='SAME')
    with tf.name_scope('initialize'):
        x_ = tf.placeholder(tf.float32, [None, 20])
        # Add a dimension to the input for multiple convolutional layers.
        x = tf.expand_dims(x_, -1)
        # Ouptuts - One hot encodings of currently sounding notes
        y_ = tf.placeholder(tf.float32, [None, 88])

    with tf.name_scope('first_conv'):
        num_filters1 = 32
        weight_vars1 = weight_variables([3,1,num_filters1])
        bias_vars1 = bias_variables([num_filters1])

        conv1 = tf.nn.relu(conv_layer(x, weight_vars1) + bias_vars1, name='InitialConvolution')
        pool1 = pooling_layer(conv1)

    with tf.name_scope('second_conv'):
        num_filters2 = 64
        weight_vars2 = weight_variables([3,num_filters1,num_filters2])
        bias_vars2 = bias_variables([num_filters2])

        conv2 = tf.nn.relu(conv_layer(pool1, weight_vars2) + bias_vars2)
        pool2 = pooling_layer(conv2)

    with tf.name_scope('fc_layer'):
        num_nodes3 = 1024
        # MFCC will be 5 after two pooling layers
        total_nodes = 5 *  num_filters2
        weight_vars3 = weight_variables([total_nodes, num_nodes3])
        bias_vars3 = bias_variables([num_nodes3])

        pool2_flat = tf.reshape(pool2, [-1, total_nodes])
        fc_layer = tf.nn.relu(tf.matmul(pool2_flat, weight_vars3) + bias_vars3)

        with tf.name_scope('dropout'):
            # Apply dropout to fully connected layer
            keep_prob = tf.placeholder(tf.float32)
            fc_layer_dropout = tf.nn.dropout(fc_layer, keep_prob)

    with tf.name_scope('readout'):
        weight_vars4 = weight_variables([num_nodes3, 88])
        bias_vars4 = bias_variables([88])

        readout = tf.matmul(fc_layer_dropout, weight_vars4) + bias_vars4

    def interpretation(logit, cutoff):
        # Returns a tensor that represents how we are interpreting the output
        # of our model.
        # |logit| is a [1, 88] tensor representing the likelihood that each
        # note is present in a given sample.
        # |cutoff| is the probability at which we will consider a note to be
        # present.
        with tf.name_scope('interpretation'):
            return tf.cast(tf.greater(logit, cutoff), tf.float32)

    with tf.Session() as sess:
        if "debug" in argv:
            sess = tf_debug.LocalCLIDebugWrapperSession(sess)
            sess.add_tensor_filter("has_inf_or_nan", tf_debug.has_inf_or_nan)

        # Don't use softmax since the outputs aren't mutually exclusive.
        loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=y_,
                logits=readout))
        with tf.name_scope('train'):
            training_step = tf.train.AdamOptimizer(1e-4).minimize(loss)

        # For now, analyze successfulness as number of predictions it gets
        # exactly right. We can revisit this later.
        with tf.name_scope('accuracy'):
            with tf.name_scope('predictions'):
                # For each note, did it get the right prediction?
                correct_predictions7 = tf.cast(tf.equal(interpretation(readout,
                    0.7), y_), tf.float32)
                correct_predictions8 = tf.cast(tf.equal(interpretation(readout,
                    0.8), y_), tf.float32)
                correct_predictions9 = tf.cast(tf.equal(interpretation(readout,
                    0.9), y_), tf.float32)

                # Did it get the right prediction for every note? Look if the
                # min value is 1.
                correct_prediction7 = tf.equal(tf.argmin(correct_predictions7,
                    1),1)
                correct_prediction8 = tf.equal(tf.argmin(correct_predictions8,
                    1),1)
                correct_prediction9 = tf.equal(tf.argmin(correct_predictions9,
                    1),1)

            accuracy7 = tf.reduce_mean(tf.cast(correct_prediction7,
                tf.float32))
            accuracy8 = tf.reduce_mean(tf.cast(correct_prediction8,
                tf.float32))
            accuracy9 = tf.reduce_mean(tf.cast(correct_prediction9,
                tf.float32))

            tf.summary.scalar('accuracy7', accuracy7)
            tf.summary.scalar('accuracy8', accuracy8)
            tf.summary.scalar('accuracy9', accuracy9)

        merged = tf.summary.merge_all()
        train_writer = tf.summary.FileWriter('/tmp/tensorboard/train', sess.graph)
        test_writer = tf.summary.FileWriter('/tmp/tensorboard/test')

        sess.run(tf.global_variables_initializer())
        for i in range(TRAINING_STEPS):
            step_data = training_data[i * BATCH_SIZE:i * BATCH_SIZE + BATCH_SIZE]
            step_labels = training_labels[i * BATCH_SIZE : i * BATCH_SIZE + BATCH_SIZE]

            if i % 10 == 0:
                #step_accuracy7 = accuracy7.eval(feed_dict={x_:step_data,
                #    y_:step_labels, keep_prob:1})
                #step_accuracy8 = accuracy8.eval(feed_dict={x_:step_data,
                #    y_:step_labels, keep_prob:1})
                #step_accuracy9 = accuracy9.eval(feed_dict={x_:step_data,
                #    y_:step_labels, keep_prob:1})
                #loss_val = loss.eval(feed_dict={x_:step_data,
                #    y_:step_labels, keep_prob:1})
                val_list = [merged, loss, accuracy7, accuracy8, accuracy9]
                summary, loss_val, step_accuracy7, step_accuracy8, step_accuracy9  = sess.run(val_list, feed_dict={x_:step_data,
                    y_:step_labels, keep_prob:1})

                print('Step: %d:\n    Accuracy 0.7: %g\n    Accuracy 0.8: %g\n    Accuracy 0.9: %g\n    Loss: %g'%(i,
                    step_accuracy7,step_accuracy8,step_accuracy9, loss_val))
                test_writer.add_summary(summary, i)
            else:
                summary, _ = sess.run([merged, training_step], feed_dict={x_:step_data,
                    y_:step_labels, keep_prob:0.5})
                train_writer.add_summary(summary, i)
            #training_step.run(feed_dict={x_:step_data, y_:step_labels,
            #    keep_prob:0.5})

    # TODO evaluate loss on validation data.

if __name__ == '__main__':
   tf.app.run()
