"""
    Loading flower photo dataset in tfrecord format to tf.Dataset, and training an inference model.

    @author: Liu Weijie
    @date: 2018-06-17
"""
import tensorflow as tf


BATCH_SIZE = 32
NUM_EPOCHS = 10
BUFFER_SIZE = 1000  # 用于shuffle tf.Dataset的内存空间，越大随机效果越好，但是越占内存
NUM_CLASSES = 5
TRAIN_DATA_FILE = "/home/jagger/workspace/tmp/datasets/flower_photos_train.tfrecord"


def parser(record):
    features = tf.parse_single_example(
        record,
        features={
            'image_raw': tf.FixedLenFeature([1, ], tf.string),
            'label': tf.FixedLenFeature([1, ], tf.int64),
            'shape': tf.FixedLenFeature([3, ], tf.int64)
        }
    )
    image = tf.decode_raw(features['image_raw'], tf.float32)[0]
    image = tf.reshape(image, features['shape'])
    label = features['label']

    return image, label


def image_augumentation(image, label):
    """
        这里可以添加一些数据增广的方法
    """
    pass
    return image, label


def train_model():

    # Create input_files
    input_files = tf.placeholder(tf.string, [None], name="input_files")

    # Create an tf.Dataset for loading data
    with tf.variable_scope("Dataset"):

        dataset = tf.data.TFRecordDataset(input_files)
        dataset = dataset.map(parser)
        dataset = dataset.map(image_augumentation)
        dataset = dataset.shuffle(buffer_size=BUFFER_SIZE).batch(BATCH_SIZE)
        dataset = dataset.repeat(NUM_EPOCHS)

        iterator = dataset.make_initializable_iterator()
        input_images, labels = iterator.get_next(name='input_images')

    with tf.Session() as sess:

        sess.run([
            tf.global_variables_initializer(),
            tf.local_variables_initializer(),
            iterator.initializer
        ], feed_dict={
            input_files: [TRAIN_DATA_FILE]
        })

        image, label = sess.run([input_images, labels])

        print(image.shape)
        print(label.shape)

        # 使用如下方式训练模型，当迭代器迭代完所有EPOCHS后，会抛出OutOfRangeError错误
        # while True:
        #     try:
        #         sess.run([train_step])
        #     except tf.errors.OutOfRangeError:
        #         break


def main(argv=None):

    train_model()


if __name__ == "__main__":
    tf.app.run()