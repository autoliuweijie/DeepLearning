"""
    An example model saving
    @author: Liu Weijie
    @date: 2018-04-09
    @reference: <Tensorflow: Deeplearning framework in Action>/Chapter5.4
"""
import tensorflow as tf

# importing graph
saver = tf.train.import_meta_graph("/home/jagger/workspace/tmp/model.ckpt.meta")

with tf.Session() as sess:

    # loading variable value to sess
    saver.restore(sess, "/home/jagger/workspace/tmp/model.ckpt")
    result = tf.get_default_graph().get_tensor_by_name("add:0")
    print(sess.run(result))
