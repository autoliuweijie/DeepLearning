# coding: utf-8
"""
    Training of a LeNet-5 approximate model on Mnist dataset.
    @author: Liu Weijie
    @date: 2018-06-10
    @ref: <Tensorflow: 实战Google深度学习框架> Chapter 6.
"""
import os
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import mnist_inference
import numpy as np


# Configuration
MNIST_DATASET_PATH = "/home/jagger/workspace/datasets/mnist_dataset/"
BATCH_SIZE = 100
LEARNING_RATE_BASE = 0.8
LEARNING_RATE_DECAY = 0.99
TRAINING_STEPS = 30000
REGULARIZER = 'L1'
REGULARAZTION_RATE = 0.0001
MOVING_AVERAGE_DECAY = 0.99
MODEL_SAVING_PATH = "/home/jagger/workspace/DeepLearning/learning_tensorflow/c06_image_recognition_and_cnn/LeNet-5/models/"
MODEL_NAME = "mnist.ckpt"
SAVING_MODEL_EVERY_STEPS = 1000


def train(mnist):

    # create input_x and y_true
    x_input = tf.placeholder(tf.float32, [None, mnist_inference.IMAGE_SIZE[0], mnist_inference.IMAGE_SIZE[1],
                                          mnist_inference.NUM_CHANNLELS], name='x_input')
    y_true = tf.placeholder(tf.float32, [None, mnist_inference.OUTPUT_NODE], name='y_true')

    regularizer = None
    if REGULARIZER == 'L1':
        regularizer = tf.contrib.layers.l1_regularizer(REGULARAZTION_RATE)
    elif REGULARIZER == 'L2':
        regularizer = tf.contrib.layers.l2_regularizer(REGULARAZTION_RATE)
    y_pred = mnist_inference.inference(x_input, train=True, regularizer=regularizer)

    # Add MovingAverage shadow node
    global_step = tf.Variable(0, trainable=False)
    variable_averages = tf.train.ExponentialMovingAverage(MOVING_AVERAGE_DECAY, global_step)
    variable_averages_op = variable_averages.apply(tf.trainable_variables())

    # define loss function
    cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=y_pred, labels=tf.argmax(y_true, 1))
    cross_entropy_mean = tf.reduce_mean(cross_entropy)
    loss = cross_entropy_mean + tf.add_n(tf.get_collection('losses'))
    
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

    # saving graph log
    log_writer = tf.summary.FileWriter("/home/jagger/tmp", tf.get_default_graph())
    log_writer.close()

    # create model saver
    saver = tf.train.Saver()

    # training model
    with tf.Session() as sess:

        tf.global_variables_initializer().run()

        for i in range(TRAINING_STEPS):

            # fetch one batch of train data
            x_batch, y_batch = mnist.train.next_batch(BATCH_SIZE)
            x_batch_reshaped = np.reshape(x_batch, (BATCH_SIZE, 
            mnist_inference.IMAGE_SIZE[0], mnist_inference.IMAGE_SIZE[1], mnist_inference.NUM_CHANNLELS))

            _, loss_value, step = sess.run([train_op, loss, global_step], feed_dict={x_input: x_batch_reshaped, y_true: y_batch})

            # saving model every 1000 steps
            if i % SAVING_MODEL_EVERY_STEPS == 0:

                print("After %d training steps, loss is %g in training batch dataset;" % (step, loss_value))
                saver.save(sess, os.path.join(MODEL_SAVING_PATH, MODEL_NAME), global_step=global_step)


def main(argc=None):

    mnist = input_data.read_data_sets(MNIST_DATASET_PATH, one_hot=True)
    train(mnist)




if __name__ == "__main__":
    tf.app.run()