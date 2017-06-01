# silence random tf warnings
import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'

import tensorflow as tf

with tf.Session() as sess:
    b = tf.Variable([3.])
    W = tf.Variable([-3.])
    x = tf.placeholder(tf.float32) # inputs 
    linear_model = W * x + b

    init = tf.global_variables_initializer()
    sess.run(init)

    y = tf.placeholder(tf.float32) # correct values
    squared_deltas = tf.square(linear_model - y)
    loss = tf.reduce_sum(squared_deltas) # just computes sum of the 1-d tensor in this case

    optimizer = tf.train.GradientDescentOptimizer(.01) # step size?
    train = optimizer.minimize(loss)

    sess.run(init) # reset values to incorrect defaults
    steps = 1000
    for i in range(steps):
        sess.run(train, {x:[1,2,3,4], y:[0,-1,-2,-3]})
    curr_w, curr_b, curr_loss = sess.run([W, b, loss], {x:[1,2,3,4], y:[0,-1,-2,-3]})
    # print("W: %s b: %s loss: %s"%(curr_w, curr_b, curr_loss))

# now with contrib

features = [tf.contrib.layers.real_values_column("x", dimension=1)]
estimator = tf.contrig.learn.LinearRegressor(feature_columns=features)
x = np.array([1., 2., 3., 4., ])
y = np.array([0., -1., -2., -3., ])
input_fn = tf.contrib.learn.io.numpy_input_fn({"x":x}, y, batch_size=4,
        num_epochs=1000)
estimator.fit(input_fn=input_fn, steps=1000)

print(estimator.evaluate(input_fn=input_fn))
