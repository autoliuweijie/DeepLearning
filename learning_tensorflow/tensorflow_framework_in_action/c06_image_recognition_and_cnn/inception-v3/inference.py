# coding: utf-8
"""
    Implementation of training, saving and testing an inception-v3 model on flower_photos dataset.
    @author: Liu Weijie
    @date: 2018-06-10
    @ref: <Tensorflow: 实战Google深度学习框架> Chapter 6.
          https://github.com/tensorflow/tensorflow/blob/master/tensorflow/contrib/slim/python/slim/nets/inception_v3.py
    @tensorflow-1.8.0
    
    @print in terminal:
        ...
        Start training inception-v3.
        [2018-06-17 00:35:45] Step 0: Validation accuracy = 20.102%
        It is the best model until now, and will be saved!
        [2018-06-17 00:38:51] Step 50: Validation accuracy = 29.771%
        It is the best model until now, and will be saved!
        [2018-06-17 00:42:17] Step 100: Validation accuracy = 45.547%
        It is the best model until now, and will be saved!
        [2018-06-17 00:45:44] Step 150: Validation accuracy = 55.725%
        It is the best model until now, and will be saved!
        [2018-06-17 00:49:12] Step 200: Validation accuracy = 56.489%
        It is the best model until now, and will be saved!
        [2018-06-17 00:52:46] Step 250: Validation accuracy = 61.832%
        It is the best model until now, and will be saved!
        [2018-06-17 00:56:11] Step 299: Validation accuracy = 58.270%
        End training.
        Final test accuracy = 63.764% (from the model at 250 step(s)).
"""
import numpy as np
import tensorflow as tf
import tensorflow.contrib.slim as slim
import tensorflow.contrib.slim.python.slim.nets.inception_v3 as inception_v3  # import inception_v3 model
import time


# Configuration
INPUT_DATA = "/home/jagger/workspace/tmp/flower_processed_data.npy"
CKPT_FILE = "/home/jagger/tmp/inception_v3.ckpt"
LOG_DIR = "/home/jagger/tmp/log/"
MAX_TO_KEEP = 5  # the maximum number models to be saved in disk.
LEARNING_RATE = 0.0001
TRAINING_STEPS = 300
PRINT_EVERY_STEPS = 50
BATCH_SIZE = 32
NUM_CLASSES = 5
SEED = 7


# Initialize
np.random.seed(SEED)
features_mean = None
features_std = None


def random_get_batch_data(features, labels, batch_size):
    """
    Get a batch of dataset randomly.
    """
    num_features = features.shape[0]
    choice = np.random.choice(num_features, batch_size)
    features_batch = features[choice]
    labels_batch = labels[choice]
    return features_batch, labels_batch


def normalize_features(features, is_train=False):
    """ Normalize features."""

    global features_mean, features_std

    if is_train:
        features_mean = np.mean(features, axis=0)
        features_std = np.std(features, axis=0)

    assert (features_mean is not None and features_std is not None), "Normaliza dataset failed!"
    features_nor = (features - features_mean) / features_std

    return features_nor


def train_save_and_test_model():

    # Loading datasets
    print("Loading dataset.")
    dataset_np = np.load(INPUT_DATA)
    training_images, training_labels, num_training_images = dataset_np[0], dataset_np[1], len(dataset_np[0])
    validation_images, validation_labels, num_validation_images = dataset_np[2], dataset_np[3], len(dataset_np[2])
    testing_images, testing_labels, num_testing_images = dataset_np[4], dataset_np[5], len(dataset_np[4])
    print("Loading dataset successfully: %s training images, %s validation images and %s testing images!" %
          (num_training_images, num_validation_images, num_testing_images))

    # Create an inception-v3 model
    print("Start creating an inception-v3 model.")
    input_images = tf.placeholder(tf.float32, [None, 299, 299, 3], name="input_images")
    with slim.arg_scope(inception_v3.inception_v3_arg_scope()):   # Default InceptionV3 arg scope
        logits, _ = inception_v3.inception_v3(input_images, num_classes=NUM_CLASSES)
    # All l2-regularization has been added to the collection of tf.GraphKeys.REGULARIZATION_LOSSES when the
    # inception-v3 be created. You can print them by the following line.
    # print(tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES))

    # Add graph for training
    labels_true = tf.placeholder(tf.int32, [None], name='labels_true')
    with tf.variable_scope("Loss"):
        softmax_cross_entropy_loss = tf.losses.softmax_cross_entropy(tf.one_hot(labels_true, NUM_CLASSES), logits)
        loss = tf.add(softmax_cross_entropy_loss, tf.add_n(tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)),
                      name="loss")

    # Adding graph for calculating acuuracy
    with tf.variable_scope("Accuracy"):
        correct_predictions = tf.equal(tf.cast(tf.argmax(logits, 1), tf.int32), labels_true)
        accuracy = tf.reduce_mean(tf.cast(correct_predictions, tf.float32))

    # Create an optimizer
    train_step_op = tf.train.RMSPropOptimizer(LEARNING_RATE).minimize(loss)

    # Create log for visualization by tensorboard
    print("Creating the inception-v3 model successfuly!")
    log_writer = tf.summary.FileWriter(LOG_DIR, graph=tf.get_default_graph())
    log_writer.close()

    # Training, evaluating and saving model
    saver = tf.train.Saver(max_to_keep=MAX_TO_KEEP)
    with tf.Session() as sess:

        init_op = tf.global_variables_initializer()
        sess.run(init_op)

        # normalize dataset
        training_images = normalize_features(training_images, is_train=True)
        validation_images = normalize_features(validation_images, is_train=False)
        testing_images = normalize_features(testing_images, is_train=False)

        # print all trainable variables
        print("The trainable variables are as follows:")
        for trainable_variable in tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES):
            print(trainable_variable)

        print("Start training inception-v3.")
        best_accuracy = 0.0
        best_step = 0
        for i in range(1, TRAINING_STEPS+1):

            images_batch, labels_batch = random_get_batch_data(training_images, training_labels, batch_size=BATCH_SIZE)

            sess.run(train_step_op, feed_dict={
                input_images: images_batch,
                labels_true: labels_batch,
            })

            if i % PRINT_EVERY_STEPS == 0 or i + 1 == TRAINING_STEPS:

                validation_accuacy = sess.run(accuracy, feed_dict={
                    input_images: validation_images,
                    labels_true: validation_labels,
                })
                now_time = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
                print("[%s] Step %d: Validation accuracy = %.3f%%" % (now_time, i, validation_accuacy*100))

                if validation_accuacy >= best_accuracy:
                    best_accuracy = validation_accuacy
                    best_step = i
                    print("It is the best model until now, and will be saved!")
                    saver.save(sess, CKPT_FILE, global_step=best_step)
        print("End training.")

        # Finally, testing the best model on the tesing datset
        saver.restore(sess, CKPT_FILE + '-%d' % (best_step))
        validation_accuacy = sess.run(accuracy, feed_dict={
            input_images: testing_images,
            labels_true: testing_labels,
        })
        print("Final test accuracy = %.3f%% (from the best model at %d step(s))." % (validation_accuacy*100, best_step))


def main(argv=None):
    train_save_and_test_model()

if __name__ == "__main__":
    tf.app.run()






