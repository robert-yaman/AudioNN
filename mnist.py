# From tutorial: https://www.tensorflow.org/get_started/mnist/beginners

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import sys

from tensorflow.examples.tutorials.mnist import input_data

import tensorflow as tf

FLAGS = None


def main(_):
  # Import data
  mnist = input_data.read_data_sets(FLAGS.data_dir, one_hot=True)

  # Create the model
  # 28 x 28 image has 784 data  points.
  # 10 sets of weights because we have an individual model for each digit
  W = tf.Variable(tf.zeros([784, 10]))
  b = tf.Variable(tf.zeros([10])) # 0-9 
  # We need 10 bias values b/c we are using one-hot encoding
  y = tf.matmul(x, W) + b

  # Define loss and optimizer
  y_ = tf.placeholder(tf.float32, [None, 10])

  # The raw formulation of cross-entropy,
  #
  #   tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(tf.nn.softmax(y)),
  #                                 reduction_indices=[1]))
  #
  # can be numerically unstable.
  #
  # So here we use tf.nn.softmax_cross_entropy_with_logits on the raw
  # outputs of 'y', and then average across the batch.
  cross_entropy = tf.reduce_mean(
      tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y))
  train_step = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)

  sess = tf.InteractiveSession()
  tf.global_variables_initializer().run()
  # Train
  for _ in range(1000):
    batch_xs, batch_ys = mnist.train.next_batch(100)
    sess.run(train_step, feed_dict={x: batch_xs, y_: batch_ys})

  # Test trained model
  correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
  accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
  print(sess.run(accuracy, feed_dict={x: mnist.test.images,
                                      y_: mnist.test.labels}))

def main_NN(_):
    def weight_variables(shape):
        # Initialized with some noise for "symmetry breaking"
        # noise introducted by initializing from normal distribution
        initial = tf.truncated_normal(shape, stddev=0.1)
        return tf.Variable(initial)
    def bias_variables(shape):
        # initialize biases tf.nn.with slightly positive wegith to avoid "dead
        # neurons"
        initial = tf.constant(0.1, shape=shape)
        return tf.Variable(initial)
    def conv2d(x, W):
        # x is 4-d tensor - batch, height, width, in_channels
            # what is batch? - num of filters we want
        # W is filter
        # (returns an entire convolutional step, not merely a layer
        return tf.nn.conv2d(x,W, strides=[1,1,1,1], padding = 'SAME')
    def max_pool_2x2(x):
        return tf.nn.max_pool(x, ksize=[1,2,2,1], strides=[1,2,2,1],
        padding='SAME')

    x = tf.placeholder(tf.float32, [None, 784])
    y_ = tf.placeholder(tf.float32, [None, 10])
    # fist convoultional layer has 5x5 filters, 1 input  channel (since images
    # are BW), and 32 is output channel (e.g. we'll make 32 channels
    # note we are making variables of this shape
    W_conv1 = weight_variables([5,5,1,32]) # confused by the order of
    # idimensions
    # Notice dimension of bias tensor - bias value is the same for each layer
    # because the weights in the filter are constant per layer
    b_conv1 = bias_variables([32])

    # 1 is number of channels
    x_image = tf.reshape(x, [-1,28,28,1])

    h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)
    h_pool1 = max_pool_2x2(h_conv1)

    # second conv layer
    # 32 because our previous conv layer output was depth of 32
    W_conv2 = weight_variables([5,5,32,64])
    b_conv2 = bias_variables([64])
    h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
    h_pool2 = max_pool_2x2(h_conv2)
    # fully connected layer
    # image size will now be 7x7 after 2 conv layers
        # with depth of 64
    W_fc1 = weight_variables([7 * 7 * 64, 1024])
    b_fc1 = bias_variables([1024])
    # flatten pool
    h_pool2_flat = tf.reshape(h_pool2, [-1, 7*7*64])
    h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)

    #apply dropout
    # (remember dropout applies to training time. Test time all nodes are
    # present
    keep_prob = tf.placeholder(tf.float32)
    h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

    # readout layer
    W_fc2 = weight_variables([1024, 10])
    b_fc2 = bias_variables([10])

    y_conv = tf.matmul(h_fc1_drop, W_fc2) + b_fc2

    # train (takes a while)
    sess = tf.InteractiveSession()
    mnist = input_data.read_data_sets(FLAGS.data_dir, one_hot=True)
    sess.run(tf.initialize_all_variables())
    # just takes the mean of the reduction of the tensro. Here we're reducing
    # cross entropy across all of the training data
    cross_entropy = tf.reduce_mean(
        tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y_conv))
    train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)
    # Why take max here? Should never be over 1
        # A: the second argument is actually the index over which we're getting
        # the max value.
    correct_prediction = tf.equal(tf.argmax(y_conv,1), tf.argmax(y_,1)) 
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    sess.run(tf.global_variables_initializer())
    for i in range(20000):
      batch = mnist.train.next_batch(50)
      if i%100 == 0:
        # this is why keep_prob is a placehodler. 
        train_accuracy = accuracy.eval(feed_dict={
            x:batch[0], y_: batch[1], keep_prob: 1.0})
        print("step %d, training accuracy %g"%(i, train_accuracy))
      train_step.run(feed_dict={x: batch[0], y_: batch[1], keep_prob: 0.5})

    print("test accuracy %g"%accuracy.eval(feed_dict={
        x: mnist.test.images, y_: mnist.test.labels, keep_prob: 1.0}))
    # result: got almost 1 accuracy after 8,000 iterations, took about 20
    # minutes


def main_NN_no_looking(_):
    def weight_variables(shape):
        initial = tf.truncated_normal(shape, stddev=.01)
        return tf.Variable(initial)
    def bias_variables(shape):
        initial = tf.constant(0.1, shape=shape)
        return tf.Variable(initial)
    def conv_layer(inputs, filter_shape):
        return tf.nn.conv2d(inputs, filter_shape, strides=[1,1,1,1],
                padding='SAME')
    def pooling_layer(inputs):
        return tf.nn.max_pool(inputs, ksize=[1,2,2,1],
                strides=[1,2,2,1], padding='SAME')
    # inputs
    # None - corresponds to variable length. Because we don't know how many
    # data points we'll have ahead of time 
    x = tf.placeholder(tf.float32, [None, 784])
    y_ = tf.placeholder(tf.float32, [None, 10])

    # -1 because we don't know ahead of time how big it wil be
    x_image = tf.reshape(x,[-1, 28,28,1])
    # First conv layer
    num_filters1 = 32
    # Q: Why is the first dimension 5 when the first dimension of the input is
    # |data|?
    # A: This filter applies over the first dimension of the input 
    weight_vars1 = weight_variables([5,5,1,num_filters1])
    bias_vars1 = bias_variables([num_filters1])

    conv1 = tf.nn.relu(conv_layer(x_image, weight_vars1) + bias_vars1)
    pool1 = pooling_layer(conv1)

    num_filters2 = 64
    weight_vars2 = weight_variables([5,5,num_filters1, num_filters2])
    bias_vars2 = bias_variables([num_filters2])

    conv2 = tf.nn.relu(conv_layer(pool1, weight_vars2) + bias_vars2)
    pool2 = pooling_layer(conv2)

    pool2_flat = tf.contrib.layers.flatten(pool2)
    # fully connected layer
    num_inputs = 7 * 7 * 64
    num_nodes = 1024

    weight_vars3 = weight_variables([num_inputs, num_nodes])
    bias_vars = bias_variables([num_nodes])
    fc_layer = tf.nn.relu(tf.matmul(pool2_flat, weight_vars3) + bias_vars)

    keep_prob = tf.placeholder(tf.float32)
    dropout = tf.nn.dropout(fc_layer, keep_prob)

    # Readout layer
    weight_vars4 = weight_variables([num_nodes, 10])
    bias_vars4 = bias_variables([10])
    readout = tf.matmul(dropout, weight_vars4) + bias_vars4

    # initilalize session
    with tf.Session() as sess:
        sess.run(tf.initialize_all_variables())
        tdata = input_data.read_data_sets(FLAGS.data_dir,
                one_hot=True)
    # define loss
        # reduce_mean is not reducing a value - its reducing a tensor to a
        # single value
        loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y_,
                logits=readout))
        training_step = tf.train.AdamOptimizer(1e-4).minimize(loss)
        correct_prediction = tf.equal(tf.argmax(readout,1), tf.argmax(y_, 1))
        accuracy = tf.reduce_mean(tf.cast(correct_prediction,tf.float32))
        sess.run(tf.global_variables_initializer())
        for i in range(10000):
            step_data = tdata.train.next_batch(50)
            if i % 100 == 0:
                step_accuracy = accuracy.eval(feed_dict={x:step_data[0], y_:step_data[1],
                        keep_prob:1.0})
                print('Step: %d Accuracy: %g'%(i, step_accuracy))
            training_step.run(feed_dict={x:step_data[0], y_:step_data[1],
                keep_prob:0.5})
    # train

    # print

if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument('--data_dir', type=str, default='/tmp/tensorflow/mnist/input_data',
                      help='Directory for storing input data')
  FLAGS, unparsed = parser.parse_known_args()
  tf.app.run(main=main_NN_no_looking, argv=[sys.argv[0]] + unparsed)

