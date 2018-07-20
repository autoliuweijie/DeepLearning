"""
    This is an simple example of LSTM for time-series prediction.
"""
import tensorflow as tf
import numpy as np
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt



LSTM_HIDDEN_SIZE = 30
LOG_DIR = "/home/jagger/workspace/tmp/log/"
TIME_STEPS = 3
NUM_FEATURES = 5
BATCH_SIZE = 32
TRAIN_STEPS = 1000


def generate_sin_data():
    """生成sin信号序列，并截取成训练和测试两部分"""

    time = np.linspace(0, 100*np.pi, 10000)
    y = np.sin(time).astype(np.float32)

    training_series = y[:8000]
    testing_series = y[8000:]

    return training_series, testing_series


def series_to_dataset(series):
    """将序列转化为可以用于训练的数据集形式"""

    features, labels = [], []
    for i in range(TIME_STEPS, len(series)-1):
        feature = series[i-TIME_STEPS: i].reshape((-1, 1))
        label = series[i+1].reshape((1))
        features.append(feature)
        labels.append(label)

    return np.array(features), np.array(labels)



def lstm_model(X, y=None, is_train=False, batch_size=BATCH_SIZE):

    with tf.variable_scope("LSTM"):
        lstm_cell = tf.nn.rnn_cell.BasicLSTMCell(LSTM_HIDDEN_SIZE)  # 这一步不会在图上创建节点
        state = lstm_cell.zero_state(batch_size, tf.float32)

        outputs = []
        for i in range(TIME_STEPS):  # 可以用tf.nn.dynamic_rnn()代替
            if i > 0: tf.get_variable_scope().reuse_variables()
            lstm_output, state = lstm_cell(X[:, i, :], state)
            outputs.append(lstm_output)

        # final_output = tf.concat(outputs, axis=1, name='output')
        final_output = outputs[-1]  # 我们只去最后一个时刻的输出

    with tf.variable_scope("Prediction"):
        predictions = tf.contrib.layers.fully_connected(final_output, 1, activation_fn=None)
        print(predictions.get_shape())

    if not is_train:
        return predictions, None, None

    loss = tf.losses.mean_squared_error(labels=y, predictions=predictions)
    train_op = tf.contrib.layers.optimize_loss(loss, tf.train.get_global_step(), optimizer='Adagrad', learning_rate=0.1)
    return predictions, loss, train_op


def training_lstm(sess, train_features, train_labels):
    ds = tf.data.Dataset.from_tensor_slices((train_features, train_labels))
    ds = ds.repeat().shuffle(10000).batch(BATCH_SIZE)
    X, y = ds.make_one_shot_iterator().get_next()

    with tf.variable_scope("model"):
        predictions, loss, train_op = lstm_model(X, y, is_train=True, batch_size=BATCH_SIZE)

    sess.run(tf.global_variables_initializer())
    for step in range(TRAIN_STEPS):
        _, loss_value = sess.run([train_op, loss])
        if step % 100 == 0:
            print("training step %s / %s: loss: %s" % (step, TRAIN_STEPS, loss_value))


def testing_lstm(sess, test_features, test_labels):

    batch_size = 1

    ds = tf.data.Dataset.from_tensor_slices((test_features, test_labels))
    ds = ds.batch(batch_size)
    X, y = ds.make_one_shot_iterator().get_next()

    with tf.variable_scope("model", reuse=True):
        predictions, _, _ = lstm_model(X, y, is_train=False, batch_size=batch_size)

    y_ps, y_ts = [], []
    while True:
        try:
            y_p, y_t = sess.run([predictions, y])
            y_ps.append(y_p[0][0])
            y_ts.append(y_t[0][0])
        except tf.errors.OutOfRangeError:
            break

    plt.figure(figsize=(14, 14))
    plt.plot(y_ps, label="predictions")
    plt.plot(y_ts, label="real values")
    plt.savefig("/home/jagger/workspace/tmp/result.png")


if __name__ == "__main__":

    training_series, testing_series = generate_sin_data()
    train_features, train_labels = series_to_dataset(training_series)
    test_features, test_labels = series_to_dataset(testing_series)

    with tf.Session() as sess:
        training_lstm(sess, train_features, train_labels)
        testing_lstm(sess, test_features, test_labels)

    log_writer = tf.summary.FileWriter(LOG_DIR, tf.get_default_graph())
    log_writer.close()
