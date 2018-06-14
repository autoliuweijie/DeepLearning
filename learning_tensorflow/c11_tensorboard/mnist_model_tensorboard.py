"""
    Using tensorboard to virtualizer model of mnist
"""
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data


tf.logging.set_verbosity(tf.logging.INFO)


# Define neural network
INPUT_NODE = 784
IMAGE_SIZE = 28
IMAGE_CHANNEL = 1
CONV1_SIZE = 5
CONV1_DEEP = 32
CONV2_SIZE = 3
CONV2_DEEP = 64
FC_SIZE = 512
OUTPUT_NODE = 10
BATCH_SIZE = 100

LOG_DIR = "/Users/liuweijie/Desktop/log/"


def variable_summaries(var):
  """Attach a lot of summaries to a Tensor (for TensorBoard visualization)."""
  with tf.name_scope('summaries'):
    mean = tf.reduce_mean(var)
    tf.summary.scalar('mean', mean)
    with tf.name_scope('stddev'):
      stddev = tf.sqrt(tf.reduce_mean(tf.square(var - mean)))
    tf.summary.scalar('stddev', stddev)
    tf.summary.scalar('max', tf.reduce_max(var))
    tf.summary.scalar('min', tf.reduce_min(var))
    tf.summary.histogram('histogram', var)


def conv_layer(input_tensor, w_shape):
    """ my conv layer"""

    conv_kernel = tf.Variable(tf.truncated_normal(w_shape, mean=0.0, stddev=0.1), dtype=tf.float32, name='conv_kernel', trainable=True)
    conv_layer = tf.nn.conv2d(input_tensor, conv_kernel, strides=[1, 1, 1, 1], padding='SAME', name='conv')
    bias = tf.Variable(tf.truncated_normal([w_shape[-1]], mean=0.0, stddev=0.1), dtype=tf.float32, name='bias', trainable=True)
    output = tf.nn.relu(tf.nn.bias_add(conv_layer, bias), name='relu')
    variable_summaries(conv_kernel)

    return output


def pooling_layer(input_tensor, k_size, strides):
    """ my pooling layer """

    output = tf.nn.max_pool(input_tensor, ksize=k_size, strides=strides, padding="SAME", name="max_pooling")

    return output


def simple_conv_net(x_image):

    # define conv_1
    with tf.name_scope('conv_1'):

        conv_1 = conv_layer(x_image, [CONV1_SIZE, CONV1_SIZE, IMAGE_CHANNEL, CONV1_DEEP])
        conv_1 = pooling_layer(conv_1, k_size=[1, 2, 2, 1], strides=[1, 2, 2, 1])

    # define conv_2
    with tf.name_scope('conv_2'):

        conv_2 = conv_layer(conv_1, [CONV1_SIZE, CONV1_SIZE, CONV1_DEEP, CONV2_DEEP])
        conv_2 = pooling_layer(conv_2, k_size=[1, 2, 2, 1], strides=[1, 2, 2, 1])

    # define fully connect layer
    with tf.name_scope('fc_1'):
        fl = tf.layers.flatten(conv_2, name='flatten')
        fl_len = fl.get_shape().as_list()[1]
        weight1 = tf.Variable(tf.truncated_normal((fl_len, FC_SIZE), mean=0.0, stddev=0.1), dtype=tf.float32, name='weight', trainable=True)
        bias = tf.Variable(tf.truncated_normal((FC_SIZE, ), mean=0.0, stddev=0.1), dtype=tf.float32, name='bias', trainable=True)
        fc_1 = tf.nn.relu(tf.nn.bias_add(tf.matmul(fl, weight1), bias))

    # define output layer
    with tf.name_scope('output'):
        fc_1_len = fc_1.get_shape().as_list()[1]
        weight2 = tf.Variable(tf.truncated_normal((fc_1_len, OUTPUT_NODE), mean=0.0, stddev=0.1), dtype=tf.float32, name='weight', trainable=True)
        bias = tf.Variable(tf.truncated_normal((OUTPUT_NODE, ), mean=0.0, stddev=0.1), dtype=tf.float32, name='bias', trainable=True)
        output = tf.nn.bias_add(tf.matmul(fc_1, weight2), bias)

    return output


def train(mnist):

    # define input layer
    x_input = tf.placeholder(name='input_layer', shape=(None, INPUT_NODE), dtype=tf.float32)
    x_image = tf.reshape(x_input, (-1, IMAGE_SIZE, IMAGE_SIZE, 1), name='reshape')

    # simple conv net
    y = simple_conv_net(x_image)

    # define loss function
    y_true = tf.placeholder(name="y_true", shape=(None, OUTPUT_NODE), dtype=tf.float32)
    with tf.name_scope("loss"):
        cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=y, labels=tf.argmax(y_true, 1))
        loss = tf.reduce_mean(cross_entropy)
        tf.summary.scalar("loss", loss)  # add an summary to loss

    # define optimizer
    with tf.name_scope("optimizer"):
        global_step = tf.Variable(0, name='global_step', dtype=tf.int32)
        learning_rate = tf.train.exponential_decay(
            0.8, global_step, mnist.train.num_examples / BATCH_SIZE, 0.99)
        train_step_op = tf.train.GradientDescentOptimizer(learning_rate) \
                        .minimize(loss, global_step=global_step)
        tf.summary.scalar('learning_rate', learning_rate)  # add an summary to learning_rate

    # export graph to tensorboard
    writer = tf.summary.FileWriter(LOG_DIR, graph=tf.get_default_graph())
    merged_summary = tf.summary.merge_all()  # merge all summary operation

    # start sess to training
    with tf.Session() as sess:

        init_op = tf.global_variables_initializer()
        sess.run(init_op)

        for i in range(1000):

            x_batch, y_btach = mnist.train.next_batch(BATCH_SIZE)

            _, loss_value, step, summary_value = sess.run([train_step_op, loss, global_step, merged_summary], feed_dict={x_input: x_batch, y_true: y_btach})

            writer.add_summary(summary_value, i)  # write summary

            if i % 100 == 0:
                print("After %s step(s) training, the loss value is %s" % (step, loss_value))

    writer.close()


def main(argv=None):

    mnist = input_data.read_data_sets("/Users/liuweijie/mnist_dataset/", one_hot=True)
    train(mnist)


if __name__ == "__main__":
    tf.app.run()
