"""
    In this script, we evaluate the model trained in mnist_train.py.
    @ahthor: Liu Weijie
    @date: 2018-06-13
    @reference: <Tensorflow: Deeplearning framework in Action>/Chapter6
"""
import time
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import mnist_inference
import mnist_train
import numpy as np


# Configuration
EVAL_EVERY_SECS = 10  # evaluate model every 10s
MNIST_DATASET_PATH = "/home/jagger/workspace/datasets/mnist_dataset/"


def evaluate(mnist):

    # create a graph
    g = tf.Graph()
    g.as_default()

    # create a model
    x_input = tf.placeholder(tf.float32, [None, mnist_inference.IMAGE_SIZE[0], mnist_inference.IMAGE_SIZE[1],
        mnist_inference.NUM_CHANNLELS], name='x_input')
    y_true = tf.placeholder(tf.float32, [None, mnist_inference.OUTPUT_NODE], name='y_true')
    y_pred = mnist_inference.inference(x_input, train=False, regularizer=None)

    # create accuracy node
    correct_predictions = tf.equal(tf.argmax(y_pred, 1), tf.argmax(y_true, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_predictions, tf.float32))

    # create an saver object fot loading model weights
    # in here, we will loading the shadow tensor weights
    variable_averages = tf.train.ExponentialMovingAverage(mnist_train.MOVING_AVERAGE_DECAY)
    variables_to_restore = variable_averages.variables_to_restore()
    saver = tf.train.Saver(variables_to_restore)

    feed_dict = {
        x_input: np.reshape(mnist.validation.images, [-1, mnist_inference.IMAGE_SIZE[0], 
            mnist_inference.IMAGE_SIZE[1], mnist_inference.NUM_CHANNLELS]),
        y_true: mnist.validation.labels,
    }
    while True:

        with tf.Session() as sess:

            # tf.train.get_checkpoint_state() will find the latest ckpt file in the given dir
            ckpt = tf.train.get_checkpoint_state(mnist_train.MODEL_SAVING_PATH)
            if ckpt and ckpt.model_checkpoint_path:

                saver.restore(sess, ckpt.model_checkpoint_path)  # restore model to session
                accuracy_value = sess.run(accuracy, feed_dict=feed_dict)
                global_step = ckpt.model_checkpoint_path.split('/')[-1].split('-')[-1]
                print("After %s training steps, accuracy is %g in validation dataset;" % (global_step, accuracy_value))

        time.sleep(EVAL_EVERY_SECS)  


def main(argv=None):
    print("start evaluation!")
    mnist = input_data.read_data_sets(MNIST_DATASET_PATH, one_hot=True)
    evaluate(mnist)



if __name__ == "__main__":

    tf.app.run()