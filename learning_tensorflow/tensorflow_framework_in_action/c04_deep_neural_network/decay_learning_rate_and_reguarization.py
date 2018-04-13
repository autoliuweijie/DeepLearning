"""
    An example of using decay learning rate and L1/L2 regularization
    @author: Liu Weijie
    @date: 2018-04-08
    @reference: <Tensorflow: Deeplearning framework in Action> / Chapter4.4
"""
import tensorflow as tf
from numpy.random import RandomState


# configuration
BATCH_SIZE = 8
DATASET_SIZE = 128
STEPS = 20000
SEED = 7
REG_TYPE = 'L1'


# Generate synthetic dataset
rdm = RandomState(1)
X = rdm.rand(DATASET_SIZE, 2)
Y = [[x1 + x2 + rdm.rand() / 10.0 - 0.05] for (x1, x2) in X]


def get_weight(shape, lamda, type='L1'):
    """
    Create an weight Variable, and put L1/L2
    regularization of it to collection.
    """

    weight = tf.Variable(tf.random_normal(shape),\
        dtype=tf.float32, trainable=True)

    if type == 'L1':
        reg_loss = tf.contrib.layers.l1_regularizer(lamda)(weight)
    elif type == 'L2':
        reg_loss = tf.contrib.layers.l2_regularizer(lamda)(weight)
    else:
        raise IOError(u"type of get_weight must be 'L1' or 'L2'")

    # add this reg_loss to 'losses' collection
    tf.add_to_collection('losses', reg_loss)

    return weight


# Create an 5-layer neural network
X_input = tf.placeholder(tf.float32, shape=(None, 2))
Y_ = tf.placeholder(tf.float32, shape=(None, 1))
layer_dimension = [2, 10, 10, 10, 1]
n_layers = len(layer_dimension)

out_layer = X_input
for i in range(n_layers - 1):

    weight = get_weight((layer_dimension[i], layer_dimension[i + 1]), 0.001, type=REG_TYPE)
    bias = tf.Variable(tf.random_normal((layer_dimension[i + 1],)), dtype=tf.float32, trainable=True)
    out_layer = tf.nn.relu(tf.matmul(out_layer, weight) + bias)

mse_loss = tf.reduce_mean(tf.square(out_layer - Y_))
tf.add_to_collection('losses', mse_loss)
loss = tf.add_n(tf.get_collection('losses'))  # total loss function


# Training neural networks
global_step = tf.Variable(0)
learning_rate = tf.train.exponential_decay(0.1, global_step, 100, 0.96,\
    staircase=True)  # inintial learning rate is 0.1, and multipy 0.96 every 100 steps
learning_step = tf.train.AdamOptimizer(learning_rate).minimize(loss, global_step=global_step)
with tf.Session() as sess:

    init_op = tf.global_variables_initializer()
    sess.run(init_op)

    for i in range(STEPS):

        start = (i * BATCH_SIZE) % DATASET_SIZE
        end = min(start + BATCH_SIZE, DATASET_SIZE)
        X_batch, Y_batch = X[start: end], Y[start: end]

        sess.run(learning_step, feed_dict={
            X_input: X_batch,
            Y_: Y_batch
        })

        if i % 1000 == 0:  # print total loss value every 1000 steps

            total_loss = sess.run(loss, feed_dict={
                X_input: X,
                Y_: Y
            })
            print("After %s steps training, the cross entropy loss is %s" % (i, total_loss))
