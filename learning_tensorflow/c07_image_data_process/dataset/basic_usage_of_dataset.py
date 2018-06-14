"""
    Basic usage of tf.data.Dataset
"""
import tensorflow as tf


input_data = [1, 2, 3, 5, 8]
dataset = tf.data.Dataset.from_tensor_slices(input_data)

# for x in dataset:
#     print(x)
LOG_DIR = "/Users/liuweijie/Desktop/log/"


iterator = dataset.make_one_shot_iterator()
x = iterator.get_next()
y = x*x

writer = tf.summary.FileWriter(LOG_DIR, graph=tf.get_default_graph())
merged_summary = tf.summary.merge_all()  # merge all summary operation
writer.close()

with tf.Session() as sess:
    for i in range(len(input_data)):
        print(sess.run(y))
