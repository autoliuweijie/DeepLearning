"""
    This is an example thar translate mnist training dataset into TFRecoed format.
"""
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import numpy as np


def _int64_feature(value_list):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=value_list))


def _bytes_feature(value_list):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=value_list))


mnist = input_data.read_data_sets(
    "/home/jagger/workspace/datasets/mnist_dataset",
    dtype=tf.uint8,
    one_hot=True
)
images = mnist.train.images
labels = mnist.train.labels
pixels = images.shape[1]
num_examples = mnist.train.num_examples


output_filename = '/home/jagger/workspace/tmp/datasets/mnist_train.tfrecords'
writer = tf.python_io.TFRecordWriter(output_filename)
for idx in range(num_examples):

    image_raw = images[idx].tostring()

    features = tf.train.Features(feature={
        'label': _int64_feature([np.argmax(labels[idx])]),
        'image_raw': _bytes_feature([image_raw]),
        'pixels': _int64_feature([pixels]),
    })
    example = tf.train.Example(features=features)

    writer.write(example.SerializeToString())

writer.close()

