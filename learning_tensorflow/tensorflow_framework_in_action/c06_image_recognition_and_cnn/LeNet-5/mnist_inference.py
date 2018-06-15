# coding: utf-8
"""
    Implementation of a LeNet-5 approximate model on Mnist dataset.
    @author: Liu Weijie
    @date: 2018-06-10
    @ref: <Tensorflow: 实战Google深度学习框架> Chapter 6.
"""
import tensorflow as tf


# Configuration of model parameters.
INPUT_NODE = 784
OUTPUT_NODE = 10

IMAGE_SIZE = (28, 28)
NUM_CHANNLELS = 1
NUM_LABELS = 10

CONV1_DEEP = 32
CONV1_SIZE = 5
CONV2_DEEP = 64
CONV2_SIZE = 5
FC_SIZE = 512  # number of node in fully-connected layer


def inference(input_tensor, train, regularizer):
    """
        Implementation of a LeNet-5 approximate model.
        @input_tensor: an input 4-D tensor of mnist images with shape (batch, size_h, size_w, channels). 
        @train: True or False represent the training process of not.
        @regularizer: type of regularizer.
    """

    # Conv1 layer
    with tf.name_scope("layer1_conv"):
        conv1_weight = tf.Variable(
            tf.truncated_normal([CONV1_SIZE, CONV1_SIZE, NUM_CHANNLELS, CONV1_DEEP], mean=0.0, stddev=0.1), 
            trainable=True, name="weight"
            )
        conv1_bias = tf.Variable(tf.truncated_normal([CONV1_DEEP], mean=0.0, stddev=0.1), trainable=True, name="bias")
        conv1 = tf.nn.conv2d(input_tensor, conv1_weight, strides=[1, 1, 1, 1], padding="SAME")
        relu1 = tf.nn.relu(tf.nn.bias_add(conv1, conv1_bias))


    # Pool1 layer
    with tf.name_scope("layer1_pool"):
        pool1 = tf.nn.max_pool(relu1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding="SAME")


    # Conv2 layer
    with tf.name_scope("layer2_conv"):
        conv2_weight = tf.Variable(
            tf.truncated_normal([CONV2_SIZE, CONV2_SIZE, CONV1_DEEP, CONV2_DEEP], mean=0.0, stddev=0.1),
            trainable=True, name="weight"
        )
        conv2_bias = tf.Variable(tf.truncated_normal([CONV2_DEEP], mean=0.0, stddev=0.1), trainable=True, name='bias')
        conv2 = tf.nn.conv2d(pool1, conv2_weight, strides=[1, 1, 1, 1], padding="SAME")
        relu2 = tf.nn.relu(tf.nn.bias_add(conv2, conv2_bias))


    # Pool2 layer
    with tf.name_scope("layer2_pool"):
        pool2 = tf.nn.max_pool(relu2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding="SAME")


    # Flatten layer
    with tf.name_scope("flatten"):
        # pool2_shape = pool2.get_shape().as_list()
        # num_nodes = pool2_shape[1] * pool2_shape[2] * pool2_shape[3]
        # flatten = tf.reshape(pool2, [pool2_shape[0], num_nodes])  # there is a bug of tf.reshape() in tf-1.8.0
        flatten = tf.contrib.layers.flatten(pool2)

    # Fully connected layer 1
    with tf.name_scope("layer3_fc"):
        flatten_nodes = flatten.get_shape().as_list()[1]
        fc1_weight = tf.Variable(
            tf.truncated_normal([flatten_nodes, FC_SIZE], mean=0.0, stddev=0.1), trainable=True, name='weight'
        )
        fc1_bias = tf.Variable(tf.truncated_normal([FC_SIZE], mean=0.0, stddev=0.1), trainable=True, name='bias')
        if regularizer != None:
            # only the weights in fc layer need to be regularized
            tf.add_to_collection('losses', regularizer(fc1_weight))
        fc1 = tf.nn.relu(tf.matmul(flatten, fc1_weight) + fc1_bias)
        if train:
            fc1 = tf.nn.dropout(fc1, 0.5)

    # Output layer
    with tf.name_scope("layer4_output"):
        output_weight = tf.Variable(
            tf.truncated_normal([FC_SIZE, OUTPUT_NODE], mean=0.0, stddev=0.1), trainable=True, name='weight'
        )
        output_bias = tf.Variable(tf.constant(0.1), name='bias')
        if regularizer != None:
            tf.add_to_collection('losses', regularizer(output_weight))
        output = tf.matmul(fc1, output_weight) + output_bias
        output = tf.nn.softmax(output)

    return output


def main(argv=None):
    
    x_input = tf.placeholder(tf.float32, [None, IMAGE_SIZE[0], IMAGE_SIZE[1], NUM_CHANNLELS], name='x-input')
    y = inference(x_input, train=True, regularizer=None)

    log_writer = tf.summary.FileWriter("/home/jagger/tmp", tf.get_default_graph())
    log_writer.close()






if __name__ == "__main__":
    tf.app.run()



