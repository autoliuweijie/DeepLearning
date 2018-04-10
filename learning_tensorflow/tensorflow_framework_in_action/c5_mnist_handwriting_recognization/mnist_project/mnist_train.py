"""
    In this script, we create and train an neural network, saving the weights to directory.
    @ahthor: Liu Weijie
    @date: 2018-04-09
    @reference: <Tensorflow: Deeplearning framework in Action>/Chapter5.5
"""
import os
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import mnist_inference


# Configuration for training
MNIST_DATASET_PATH = "/Users/liuweijie/mnist_dataset/"
BATCH_SIZE = 100
LEARNING_RATE_BASE = 0.8
LEARNING_RATE_DECAY = 0.99
TRAINING_STEPS = 30000
REGULARIZER = 'L1'
REGULARAZTION_RATE = 0.0001
MOVING_AVERAGE_DECAY = 0.99
MODEL_SAVING_PATH = "/Users/liuweijie/Desktop/MNIST_MODEL/"
MODEL_NAME = "mnist.ckpt"
SAVING_MODEL_EVERY_STEPS = 1000


def train(mnist):

    # Create input_tensor and y_
    x_input = tf.placeholder(tf.float32, [None, mnist_inference.INPUT_NODE], name='x_input')
    y_ = tf.placeholder(tf.float32, [None, mnist_inference.OUTPUT_NODE], name='y_true')

    y = mnist_inference.inference(x_input, regularizer='L1', lamda=REGULARAZTION_RATE)

    # Add MovingAverage shadow node
    global_step = tf.Variable(0, trainable=False)
    variable_averages = tf.train.ExponentialMovingAverage(MOVING_AVERAGE_DECAY, global_step)
    variable_averages_op = variable_averages.apply(tf.trainable_variables())

    # define loss function
    cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=y, labels=tf.argmax(y_, 1))
    cross_entropy_mean = tf.reduce_mean(cross_entropy)
    loss = cross_entropy_mean + tf.add_n(tf.get_collection(mnist_inference.LOSSES_COLLECTION))

    # define optimizer
    learning_rate = tf.train.exponential_decay(
        LEARNING_RATE_BASE,
        global_step,
        mnist.train.num_examples / BATCH_SIZE,
        LEARNING_RATE_DECAY
    )
    # train_step = tf.train.AdamOptimizer(learning_rate).minimize(loss, global_step=global_step)
    train_step = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss, global_step=global_step)
    train_op = tf.group(train_step, variable_averages_op)

    # create saver
    saver = tf.train.Saver()

    # training model
    with tf.Session() as sess:

        tf.global_variables_initializer().run()

        for i in range(TRAINING_STEPS):

            # fetch one batch of train data
            x_batch, y_batch = mnist.train.next_batch(BATCH_SIZE)

            _, loss_value, step = sess.run([train_op, loss, global_step], feed_dict={x_input: x_batch, y_: y_batch})

            # saving model every 1000 steps
            if i % SAVING_MODEL_EVERY_STEPS == 0:

                print("After %d training steps, loss is %g in training batch dataset;" % (step, loss_value))
                saver.save(sess, os.path.join(MODEL_SAVING_PATH, MODEL_NAME), global_step=global_step)


def main(argc=None):

    mnist = input_data.read_data_sets("/Users/liuweijie/mnist_dataset/", one_hot=True)
    train(mnist)


if __name__ == "__main__":
    tf.app.run()
