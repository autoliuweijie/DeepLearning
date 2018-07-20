# coding: utf-8
"""
    This is an example for loading dataset from tfrecord format.
"""
import tensorflow as tf


TFRECORD_FILE = "/home/jagger/workspace/tmp/datasets/mnist_train.tfrecords"


# 创建文件名队列
filename_quene = tf.train.string_input_producer([TFRECORD_FILE], shuffle=True)
# filename = filename_quene.dequeue()


# 使用TFRecorfReader读取一个数据, 注意serialized_example是tensor变量
reader = tf.TFRecordReader()
_, serialized_example = reader.read(filename_quene)


# 解析读取到的数据
example = tf.parse_single_example(
    serialized_example,
    features={
        'pixels': tf.FixedLenFeature([1,], tf.int64), # [1,] is shape of this feature
        'label': tf.FixedLenFeature([1,], tf.int64),
        'image_raw': tf.FixedLenFeature([1,], tf.string)
    }
)
image = tf.decode_raw(example['image_raw'], tf.uint8)[0]  # 可以将string的tensor解析成数组的tensor
label = tf.cast(example['label'], tf.int32)[0]
pixels = tf.cast(example['pixels'], tf.int32)[0]


with tf.Session() as sess:

    # start queues in tf.GraphKeys,QUEUE_RUNNERS by multi-threading
    # 如果使用了队列，以下语句是必须的
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(sess=sess, coord=coord)

    image_value, label_value = sess.run([image, label])
    print(image_value)
    print(label_value)

    # 停止多有线程
    coord.request_stop()
    coord.join(threads)