"""
    Split flower photos dataset into training, validation and testing dataset, and saved as tfrecord format.

    @author: Liu Weijie
    @date: 2018-06-17
"""
import tensorflow as tf
import os
import cv2
import numpy as np


# Configuation
INPUT_DATA = "/home/jagger/workspace/datasets/flower_photos"
OUTPUT_FILE = "/home/jagger/workspace/tmp/datasets/flower_photos_%s.tfrecord"

VALIDATION_PERCENTAGE = 10  # percentage of validation dataset
TEST_PERCENTAGE = 10  # percentage of test dataset
EXTENSIONS = ['jpg', 'jpeg', 'JPG', 'JPEG']  # extension name of images
OUT_IMAGE_SIZE = (299, 299)
SEED = 7  # random seed


def _int64_feature(value_list):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=value_list))


def _bytes_feature(value_list):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=value_list))


def image_to_example(image, label):

    image_raw = image.tostring()
    height, width, channals = image.shape

    features = tf.train.Features(feature={
        'image_raw': _bytes_feature([image_raw]),
        'label': _int64_feature([label]),
        'shape': _int64_feature([height, width, channals])
    })
    example = tf.train.Example(features=features)

    return example


def loading_and_saving_images():

    train_writer = tf.python_io.TFRecordWriter(OUTPUT_FILE % "train")
    eval_writer = tf.python_io.TFRecordWriter(OUTPUT_FILE % "eval")
    test_writer = tf.python_io.TFRecordWriter(OUTPUT_FILE % "test")

    flower_names = os.listdir(INPUT_DATA)
    flower_label = 0
    for flower_name in flower_names:

        flower_path = os.path.join(INPUT_DATA, flower_name)
        if not os.path.isdir(flower_path):
            continue

        for image in os.listdir(flower_path):

            extension = image.split('.')[-1]
            if extension not in EXTENSIONS:
                continue

            image_path = os.path.join(flower_path, image)
            image_np = cv2.cvtColor(cv2.imread(image_path, cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)
            image_np = cv2.resize(image_np, OUT_IMAGE_SIZE).astype(np.float32)

            example = image_to_example(image_np, flower_label)

            # Assign this image into training, testing or validation dataset randomly.
            chance = np.random.randint(100)
            if chance < VALIDATION_PERCENTAGE:
                eval_writer.write(example.SerializeToString())
            elif chance < (TEST_PERCENTAGE + VALIDATION_PERCENTAGE):
                test_writer.write(example.SerializeToString())
            else:
                train_writer.write(example.SerializeToString())

        flower_label += 1

    train_writer.close()
    eval_writer.close()
    test_writer.close()


def main(argv=None):
    loading_and_saving_images()
    print("All image have been saved at:")
    print(OUTPUT_FILE % "train")
    print(OUTPUT_FILE % "eval")
    print(OUTPUT_FILE % "test")


if __name__ == "__main__":
    tf.app.run()