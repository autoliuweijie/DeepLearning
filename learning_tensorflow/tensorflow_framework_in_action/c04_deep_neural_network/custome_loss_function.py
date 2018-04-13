"""
    An example of customing your loss funtion in a neural network
    @author: Liu Weijie
    @date: 2018-04-03
    @reference: <Tensorflow: Deeplearning framework in Action>/Chapter4.2
"""
import tensorflow as tf
from numpy.random import RandomState


# configuration
BATCH_SIZE = 8
DATASET_SIZE = 128
STEPS = 5000
SEED = 7


# Generate synthetic dataset
rdm = RandomState(1)
X = rdm.rand(DATASET_SIZE, 2)
Y = [[x1 + x2 + rdm.rand() / 10.0 - 0.05] for (x1, x2) in X]


# Designe an neural network
# Design structure
X_input = tf.placeholder(dtype=tf.float32, shape=(None, 2), name='X-input')
W1 = tf.Variable(tf.random_normal([2, 1], stddev=1, seed=SEED), \
    dtype=tf.float32, name='W1')
b = tf.Variable(0.0, dtype=tf.float32, name='b')
Y_output = tf.matmul(X_input, W1) + b

# Define loss function
# loss = sum(f(y, y_))
# where f(x, y) = 10(x - y) if x >= y, else f(x, y) = 1(x - y)
Y_ = tf.placeholder(shape=(None, 1), dtype=tf.float32, name='Y-true')
loss = tf.reduce_sum(tf.where(tf.greater(Y_output, Y_), (Y_output - Y_) * 10, (Y_ - Y_output)))


# Training neural neetwork
optimizer = tf.train.AdamOptimizer(0.001).minimize(loss)
with tf.Session() as sess:

    init_op = tf.global_variables_initializer()
    sess.run(init_op)

    # print w1 and b before training
    print("=====w1 and b before training:=====")
    print(sess.run(W1))
    print(sess.run(b))
    print("====================================\n")

    for i in range(STEPS):

        start = (i * BATCH_SIZE) % DATASET_SIZE
        end = min(start + BATCH_SIZE, DATASET_SIZE)
        X_batch, Y_batch = X[start: end], Y[start: end]

        sess.run(optimizer, feed_dict={
            X_input: X_batch,
            Y_: Y_batch
        })

        if i % 1000 == 0:  # print total loss value every 1000 steps

            total_loss = sess.run(loss, feed_dict={
                X_input: X,
                Y_: Y
            })

            print("After %s steps training, the cross entropy loss is %s" \
                % (i, total_loss))

    # print w1 and b after training
    print("=====w1 and b after training:=====")
    print(sess.run(W1))
    print(sess.run(b))
    print("====================================\n")
