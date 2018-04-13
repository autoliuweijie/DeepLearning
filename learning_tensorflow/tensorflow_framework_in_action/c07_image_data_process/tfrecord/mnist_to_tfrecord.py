"""
    This is an example thar translate mnist dataset into TFRecoed format.
"""
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import numpy as np


def _int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


def _bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


mnist = input_data.read_data_sets(
    "/Users/liuweijie/mnist_dataset/",
    dtype=tf.uint8,
    one_hot=True
)
images = mnist.train.images
labels = mnist.train.labels
pixels = images.shape[1]
num_examples = mnist.train.num_examples


output_filename = '/Users/liuweijie/Desktop/test/mnist.tfrecords'
writer = tf.python_io.TFRecordWriter(output_filename)
for idx in range(num_examples):

    image_raw = images[idx].tostring()

    features = tf.train.Features(feature={
        'pixels': _int64_feature(pixels),
        'label': _int64_feature(np.argmax(labels[idx])),
        'image_raw': _bytes_feature(image_raw)
    })
    example = tf.train.Example(features=features)

    writer.write(example.SerializeToString())

writer.close()


# x = tf.train.Int64List(value=[1, 2])
# print(type(x))
