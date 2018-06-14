"""
    In this script, we evaluate the model trained in mnist_train.py.
    @ahthor: Liu Weijie
    @date: 2018-04-10
    @reference: <Tensorflow: Deeplearning framework in Action>/Chapter5.5
"""
import time
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import mnist_inference
import mnist_train


# Configuration
EVAL_EVERY_SECS = 10  # evaluate model every 10s


def evaluate(mnist):

    # create a graph
    g = tf.Graph()
    g.as_default()

    # creat model
    x_input = tf.placeholder(tf.float32, [None, mnist_inference.INPUT_NODE], name='x_input')
    y_ = tf.placeholder(tf.float32, [None, mnist_inference.OUTPUT_NODE], name='y_true')
    y = mnist_inference.inference(x_input)

    # create accuray node
    correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    # create saver object to loading weight
    # In here, we will loading the shadow tensor
    variable_averages = tf.train.ExponentialMovingAverage(mnist_train.MOVING_AVERAGE_DECAY)
    variables_to_restore = variable_averages.variables_to_restore()
    saver = tf.train.Saver(variables_to_restore)

    feed_dict = {
        x_input: mnist.validation.images,
        y_: mnist.validation.labels
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
    mnist = input_data.read_data_sets("/Users/liuweijie/mnist_dataset/", one_hot=True)
    evaluate(mnist)


if __name__ == "__main__":
    tf.app.run()
