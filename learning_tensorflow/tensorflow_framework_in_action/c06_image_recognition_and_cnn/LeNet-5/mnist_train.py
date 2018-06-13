# coding: utf-8
"""
    Training of a LeNet-5 approximate model on Mnist dataset.
    @author: Liu Weijie
    @date: 2018-06-10
    @ref: <Tensorflow: 实战Google深度学习框架> Chapter 6.
"""
import os
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import mnist_inference


# Configuration
MNIST_DATASET_PATH = "/home/jagger/workspace/datasets/mnist_dataset/"
BATCH_SIZE = 100
LEARNING_RATE_BASE = 0.8
LEARNING_RATE_DECAY = 0.99
TRAINING_STEPS = 30000
REGULARIZER = 'L1'
REGULARAZTION_RATE = 0.0001
MOVING_AVERAGE_DECAY = 0.99
MODEL_SAVING_PATH = "/Volumes/jagger/workspace/DeepLearning/learning_tensorflow/" \
                    "tensorflow_framework_in_action/c06_image_recognition_and_CNN/LeNet-5/models"
MODEL_NAME = "mnist.ckpt"
SAVING_MODEL_EVERY_STEPS = 1000


mnist = input_data.read_data_sets(MNIST_DATASET_PATH, one_hot=True)