from Framework.Network import Layer
from Framework.Network.layers import Locality0
from Framework.Network.utils import make_expand
import tensorflow as tf

d = 3
inputs = tf.placeholder(tf.float32, shape=(1,4,3))
expand = make_expand(3)
print(expand.shape)

l = Layer(Locality0, inputs, tf.nn.relu, batchsize = 1,
        params = {'d': d}, name="test")
out = l.outputs()
print(out.get_shape().as_list())

with tf.Session() as sess:
    writer = tf.summary.FileWriter('testlogs', sess.graph)
    writer.close()
