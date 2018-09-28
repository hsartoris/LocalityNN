import tensorflow as tf
from Framework.Network.util.dataset import load_dataset

d = load_dataset("tfrecords_test", "train", 1, 2)
print(d)

iterator = d.make_initializable_iterator()
next_item = iterator.get_next()
batch_data = tf.reshape(next_item[0], [-1, 10, 5])
batch_labels = tf.reshape(next_item[1], [-1, 5, 5])

with tf.Session() as sess:
    sess.run(iterator.initializer)
    print(sess.run([next_item, batch_data, batch_labels]))

