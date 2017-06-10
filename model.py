import tensorflow as tf
import numpy


# TODO: make number of MFCC units a parameter, since we will likely change
def _variable_summaries(var):
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

def get_transcription_model(examples):
    def weight_variables(shape):
        with tf.name_scope('weights'):
            initial = tf.truncated_normal(shape, stddev=.01)
            variables = tf.Variable(initial)
            _variable_summaries(variables)
            return variables
    def bias_variables(shape):
        with tf.name_scope('biases'):
            initial = tf.constant(0.1, shape=shape)
            variables = tf.Variable(initial)
            _variable_summaries(variables)
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
        # Add a dimension to the input for multiple convolutional layers.
        x = tf.cast(tf.expand_dims(examples, -1), tf.float32)

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
        # TODO: Do I need to take sigmoid here since I'm not using softmax?
            # Note that my loss function takes sigmoid
        weight_vars4 = weight_variables([num_nodes3, 88])
        bias_vars4 = bias_variables([88])

        readout = tf.matmul(fc_layer_dropout, weight_vars4) + bias_vars4
        tf.summary.histogram('histogram', readout)

    return readout, keep_prob
