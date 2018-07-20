# coding: utf-8
"""
    Prepare flower_photos dataset into numpy format.

    Running this script to transform the flower_photos dataset to numpy format, and split it into training, validation,
    and test sets.

    the flower_photos dataset can be downloaded by:

        $ wget http://download.tensorflow.org/example_images/flower_photos.tgz
        $ tar xzf flower_photos.tgz

    @author: Liu Weijie
    @date: 2018-06-13
    @ref: <Tensorflow: 实战Google深度学习框架> Chapter 6.
"""
import os
import numpy as np
import cv2
import tensorflow as tf



# Configuration
INPUT_DATA = "/home/jagger/workspace/datasets/flower_photos"
OUTPUT_FILE = "/home/jagger/workspace/tmp/flower_processed_data.npy"

VALIDATION_PERCENTAGE = 10  # percentage of validation dataset
TEST_PERCENTAGE = 10  # percentage of test dataset
EXTENSIONS = ['jpg', 'jpeg', 'JPG', 'JPEG']  # extension name of images
OUT_IMAGE_SIZE = (299, 299)
SEED = 7  # random seed


# Initialize
np.random.seed(SEED)



def create_image_lists(testing_percentage, validation_percentage):
    """
    Loading dataset and split it into train, test and validation dataset, and saving these dataset in format of npy.
    :param testing_percentage: int - Percentage of validation dataset
    :param validation_percentage: int - Percentage of test dataset
    :return out_dataset: np.array - a np format dataset consit of training, testing and validation dataset.
    """

    training_images, training_labels = [], []
    testing_images, testing_labels = [], []
    validation_images, validation_labels = [], []

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

            # Assign this image into training, testing or validation dataset randomly.
            chance = np.random.randint(100)
            if chance < validation_percentage:
                validation_images.append(image_np)
                validation_labels.append(flower_label)
            elif chance < (testing_percentage + validation_percentage):
                testing_images.append(image_np)
                testing_labels.append(flower_label)
            else:
                training_images.append(image_np)
                training_labels.append(flower_label)

        flower_label += 1

    # Shuffle the training dataset
    state = np.random.get_state()
    np.random.shuffle(training_images)
    np.random.set_state(state)
    np.random.shuffle(training_labels)

    print("There are %s training image, %s validation images and %s testing images." %
          (len(training_labels), len(validation_labels), len(testing_labels)))

    out_dataset = np.asarray([np.array(training_images), np.array(training_labels),
                              np.array(validation_images), np.array(validation_labels),
                              np.array(testing_images), np.array(testing_labels)])

    return out_dataset


def main(argv=None):
    print("Start to transfer and split flower_photo dataset.")
    assert os.path.exists(INPUT_DATA), "$s doesn't exist!" % (INPUT_DATA)
    dataset = create_image_lists(VALIDATION_PERCENTAGE, TEST_PERCENTAGE)
    np.save(OUTPUT_FILE, dataset)
    print("NPY format dataset has been saved at %s, successfuly!" % (OUTPUT_FILE))


if __name__ == "__main__":
    main()
