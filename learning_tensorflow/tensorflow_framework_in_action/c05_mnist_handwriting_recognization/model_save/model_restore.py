"""
    An example model saving
    @author: Liu Weijie
    @date: 2018-04-09
    @reference: <Tensorflow: Deeplearning framework in Action>/Chapter5.4
"""
import tensorflow as tf

v1 = tf.get_variable("other-v1", shape=[1], initializer=tf.random_normal_initializer(mean=0.0, stddev=1.0), dtype=tf.float32)
v2 = tf.get_variable("other-v2", shape=[1], initializer=tf.random_normal_initializer(mean=0.0, stddev=1.0), dtype=tf.float32)
result = v1 + v2

saver = tf.train.Saver({"v1": v1, "v2": v2})

with tf.Session() as sess:

    saver.restore(sess, "/home/jagger/workspace/tmp/model.ckpt")
    print(sess.run(result))
