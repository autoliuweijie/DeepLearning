# coding: utf-8
"""
    Finetuning an inpception-v3 model from the ImageNet dataset on the flower_photos dataset.

    Download the weights file of the inception-v3 trained on ImageNet by

        $ wget http://download.tensorflow.org/models/inception_v3_2016_08_28.tar.gz
        $ tar xzf inception_v3_2016_08_28.tar.gz

    @author: Liu Weijie
    @date: 2018-06-10
    @ref: <Tensorflow: 实战Google深度学习框架> Chapter 6.
          https://github.com/tensorflow/tensorflow/blob/master/tensorflow/contrib/slim/python/slim/nets/inception_v3.py
    @tensorflow-1.8.0

    @print in terminal:
        ...
        Start training inception-v3.
        [2018-06-17 22:27:01] Step 50: Validation accuracy = 25.700%
        It is the best model until now, and saved at /home/jagger/workspace/tmp/finetune_inception_v3.ckpt-50.
        [2018-06-17 22:27:21] Step 100: Validation accuracy = 35.115%
        It is the best model until now, and saved at /home/jagger/workspace/tmp/finetune_inception_v3.ckpt-100.
        [2018-06-17 22:27:41] Step 150: Validation accuracy = 49.873%
        It is the best model until now, and saved at /home/jagger/workspace/tmp/finetune_inception_v3.ckpt-150.
        [2018-06-17 22:28:16] Step 200: Validation accuracy = 60.051%
        It is the best model until now, and saved at /home/jagger/workspace/tmp/finetune_inception_v3.ckpt-200.
        [2018-06-17 22:29:03] Step 250: Validation accuracy = 62.850%
        It is the best model until now, and saved at /home/jagger/workspace/tmp/finetune_inception_v3.ckpt-250.
        [2018-06-17 22:30:09] Step 300: Validation accuracy = 68.448%
        It is the best model until now, and saved at /home/jagger/workspace/tmp/finetune_inception_v3.ckpt-300.
        [2018-06-17 22:31:27] Step 350: Validation accuracy = 70.738%
        It is the best model until now, and saved at /home/jagger/workspace/tmp/finetune_inception_v3.ckpt-350.
        [2018-06-17 22:32:46] Step 400: Validation accuracy = 75.318%
        It is the best model until now, and saved at /home/jagger/workspace/tmp/finetune_inception_v3.ckpt-400.
        [2018-06-17 22:34:05] Step 450: Validation accuracy = 79.898%
        It is the best model until now, and saved at /home/jagger/workspace/tmp/finetune_inception_v3.ckpt-450.
        [2018-06-17 22:35:24] Step 500: Validation accuracy = 78.880%
        [2018-06-17 22:36:44] Step 550: Validation accuracy = 75.827%
        [2018-06-17 22:38:03] Step 599: Validation accuracy = 79.644%
        [2018-06-17 22:38:19] Step 600: Validation accuracy = 81.425%
        It is the best model until now, and saved at /home/jagger/workspace/tmp/finetune_inception_v3.ckpt-600.
        End training.
        Final test accuracy = 78.090% (from the best model at 600 step(s)).
"""
import numpy as np
import tensorflow as tf
import tensorflow.contrib.slim as slim
import tensorflow.contrib.slim.python.slim.nets.inception_v3 as inception_v3  # import inception_v3 model
import time


# Configuration
INPUT_DATA = "/home/jagger/workspace/tmp/flower_processed_data.npy"
CKPT_FILE = "/home/jagger/workspace/tmp/finetune_inception_v3.ckpt"  # The output ckpt file path
PRETRAIN_CKPT_FILE = "/home/jagger/workspace/tmp/inception_v3.ckpt"  # weights file of ipv3 pre-trained on imagenet
LOG_DIR = "/home/jagger/tmp/log/"

MAX_TO_KEEP = 5  # the maximum number models to be saved in disk.
LEARNING_RATE = 0.0001
TRAINING_STEPS = 600
PRINT_EVERY_STEPS = 50
BATCH_SIZE = 32
NUM_CLASSES = 5
SEED = 7

# Variables under these scopes doesn't need to be imported
CHECKPOINT_EXCLUDE_SCOPES = ['InceptionV3/Logits/', 'InceptionV3/AuxLogits/']
TRAINABLE_SCOPES = ['InceptionV3/Logits', 'InceptionV3/AuxLogits']  # Variables need to be trained.


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


def get_tuned_variables():
    """ Get model variables need to be imported. """
    all_variables = slim.get_model_variables()
    exclude_variables = []
    for exclude_scope in CHECKPOINT_EXCLUDE_SCOPES:
        exclude_variables += slim.get_model_variables(scope=exclude_scope)
    variables_to_restore = [v for v in all_variables if v not in set(exclude_variables)]

    return variables_to_restore

def get_scope_trainable_variables():
    """ Get the variables need to be retrained. """
    trainable_variables = []
    for scope in TRAINABLE_SCOPES:
        trainable_variables += tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=scope)
    return trainable_variables


def finetune_inception_v3():

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
    scope_trainable_variables = get_scope_trainable_variables()
    train_step_op = tf.train.RMSPropOptimizer(LEARNING_RATE).minimize(loss, var_list=scope_trainable_variables)  # only train the variables under the TRAINABLE_SCOPES
    # train_step_op = tf.train.RMSPropOptimizer(LEARNING_RATE).minimize(loss)  # training all trainable variables

    # Create log for visualization by tensorboard
    print("Creating the inception-v3 model successfuly!")
    log_writer = tf.summary.FileWriter(LOG_DIR, graph=tf.get_default_graph())
    log_writer.close()

    # Function for loading variables in ckpt file
    load_fn = slim.assign_from_checkpoint_fn(
        model_path=PRETRAIN_CKPT_FILE,
        var_list=get_tuned_variables(),
        ignore_missing_vars=True
    )

    # Training, evaluating and saving model
    saver = tf.train.Saver(max_to_keep=MAX_TO_KEEP)
    with tf.Session() as sess:

        init_op = tf.global_variables_initializer()
        sess.run(init_op)

        # loading pretrained variables
        print("Loading tuned variables from %s." % (PRETRAIN_CKPT_FILE))
        load_fn(sess)

        # normalize dataset
        training_images = normalize_features(training_images, is_train=True)
        validation_images = normalize_features(validation_images, is_train=False)
        testing_images = normalize_features(testing_images, is_train=False)

        # print all trainable variables
        print("The trainable variables are as follows:")
        for trainable_variable in scope_trainable_variables:
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
                print("[%s] Step %d/%d: Validation accuracy = %.3f%%" % (now_time, i, TRAINING_STEPS, validation_accuacy * 100))

                if validation_accuacy >= best_accuracy:
                    best_accuracy = validation_accuacy
                    best_step = i
                    saved_model_path = saver.save(sess, CKPT_FILE, global_step=best_step)
                    print("It is the best model until now, and saved at %s." % (saved_model_path))
        print("End training, and start testing on the testing dataset.")

        # Finally, testing the best model on the tesing datset
        saver.restore(sess, CKPT_FILE + '-%d' % (best_step))
        validation_accuacy = sess.run(accuracy, feed_dict={
            input_images: testing_images,
            labels_true: testing_labels,
        })
        print("Final test accuracy = %.3f%% (from the best model at %d step(s))." % (validation_accuacy*100, best_step))



def main(argv=None):
    finetune_inception_v3()


if __name__ == "__main__":
    tf.app.run()


