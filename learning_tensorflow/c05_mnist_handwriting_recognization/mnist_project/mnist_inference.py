"""
    In this script, we define an neural nework usd for recognize mnist handwriting.
    @ahthor: Liu Weijie
    @date: 2018-04-09
    @reference: <Tensorflow: Deeplearning framework in Action>/Chapter5.5
"""
import tensorflow as tf


# Define neural network
INPUT_NODE = 784
OUTPUT_NODE = 10
HIDEN_NODE = [500, ]
LOSSES_COLLECTION = 'losses'


# Initialize
hiden_layers = len(HIDEN_NODE)


def get_weight(shape, regularizer=None, lamda=0.0001):
    """Create an Variable tensor and put regularization to loss function collection!"""

    weight = tf.get_variable("weight", shape, initializer=tf.truncated_normal_initializer(stddev=0.1))

    if regularizer == 'L1':
        tf.add_to_collection(LOSSES_COLLECTION, tf.contrib.layers.l1_regularizer(lamda)(weight))
    elif regularizer == 'L2':
        tf.add_to_collection(LOSSES_COLLECTION, tf.contrib.layers.l2_regularizer(lamda)(weight))
    elif regularizer is None:
        pass
    else:
        raise IndexError("Error: regularizer in get_weight() must be 'L1', 'L2' or None!")

    return weight


def inference(input_tensor, regularizer=None, lamda=0.0001):
    """
        Greate an forward neutal network
    """

    # create hidden layer
    cur_node = INPUT_NODE
    cur_layer = input_tensor
    for layer_idx in range(0, hiden_layers):

        with tf.variable_scope("hidden_layer_%s" % (layer_idx + 1), reuse=False):

            out_node = HIDEN_NODE[layer_idx]
            weight = get_weight([cur_node, out_node], regularizer, lamda)
            bias = tf.get_variable("bias", [out_node])
            cur_layer = tf.nn.relu(tf.matmul(cur_layer, weight) + bias)
            cur_node = out_node

    # create output layer
    with tf.variable_scope('out_layer', reuse=False):
        weight = get_weight([cur_node, OUTPUT_NODE], regularizer, lamda)
        bias = tf.get_variable("bias", [OUTPUT_NODE])
        out_layer = tf.matmul(cur_layer, weight) + bias

    return out_layer


if __name__ == "__main__":

    input_tensor = tf.get_variable("input_layer", shape=(1000, INPUT_NODE), initializer=tf.random_normal_initializer(mean=0.0, stddev=1.0))
    y = inference(input_tensor, regularizer='L2', lamda=0.0001)

    with tf.Session() as sess:
        init_op = tf.global_variables_initializer()
        sess.run(init_op)

        print(sess.run(y))
