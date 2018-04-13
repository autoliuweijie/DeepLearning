"""
    This is an simple example of tensorboard.
"""
import tensorflow as tf


# define a computal graph
with tf.name_scope("input1"):
    input1 = tf.constant([1.0, 2.0, 3.0], name='input1')

with tf.name_scope("input2"):
    input2 = tf.Variable(tf.random_uniform([3]), name="input2")

output = tf.add_n([input1, input2], name='add')


# saving graph log
log_writer = tf.summary.FileWriter("/Users/liuweijie/Desktop/log", tf.get_default_graph())
log_writer.close()
