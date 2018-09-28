import tensorflow as tf
from Framework.Network.util.dataset import load_to_input_fn


next_item = load_to_input_fn("tfrecords_test", "train", 1, 2)
batch_data = tf.reshape(next_item[0]['time_series'], [-1, 10, 5])
batch_labels = tf.reshape(next_item[1], [-1, 5, 5])

with tf.Session() as sess:
    print(sess.run([next_item, batch_data, batch_labels]))

