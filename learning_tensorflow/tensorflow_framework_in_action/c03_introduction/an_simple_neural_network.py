"""
    An simple of a neural network
    @author: Liu Weijie
    @date: 2018-04-03
    @reference: <Tensorflow: Deeplearning framework in Action>/Chapter3
"""
import tensorflow as tf
from numpy.random import RandomState  # used for generate synthetic dataset


# configuration
batch_size = 8
seed = 1
train_steps = 5000


# create neural network
# create structure
x = tf.placeholder(tf.float32, shape=(None, 2), name='x-input')
w1 = tf.Variable(tf.random_normal([2, 3], stddev=1, seed=seed))
w2 = tf.Variable(tf.random_normal([3, 1], stddev=1, seed=seed))
y = tf.matmul(tf.matmul(x, w1), w2)

# define loss function
y = tf.sigmoid(y)
y_ = tf.placeholder(tf.float32, shape=(None, 1), name='y-input')
cross_entropy = -tf.reduce_mean(
    y_ * tf.log(tf.clip_by_value(y, 1e-10, 1.0)) +
    (1 - y) * tf.log(tf.clip_by_value(1 - y, 1e-10, 1.0))
)

# create optimizer
optimizer = tf.train.AdamOptimizer(0.001).minimize(cross_entropy)


# Generate a synthetic dataset
rdm = RandomState(1)
dataset_size = 128
X = rdm.rand(dataset_size, 2)
Y = [[int(x1 + x2 < 1)] for (x1, x2) in X]


# create a session to run neural network
with tf.Session() as sess:

    # initialize variables
    init_op = tf.global_variables_initializer()
    sess.run(init_op)

    # print w1 and w2 before training
    print("=====w1 and w2 before training:=====")
    print(sess.run(w1))
    print(sess.run(w2))
    print("====================================\n")

    for i in range(train_steps):

        start = (i * batch_size) % dataset_size
        end = min(start + batch_size, dataset_size)
        X_batch, Y_batch = X[start: end], Y[start: end]

        sess.run(optimizer, feed_dict={
            x: X_batch,
            y_: Y_batch
        })

        if i % 1000 == 0:  # print total loss value every 1000 steps

            total_loss = sess.run(cross_entropy, feed_dict={
                x: X,
                y_: Y
            })

            print("After %s steps training, the cross entropy loss is %s" \
                % (i, total_loss))

    # print w1 and w2 after training
    print("=====w1 and w2 after training:=====")
    print(sess.run(w1))
    print(sess.run(w2))
    print("====================================\n")
