import tensorflow as tf
import numpy as np
from Framework.Network import Network
from Framework.Network.Stack import Stack
from Framework.Network.layers import Locality0, Locality1, Flatten

d = 3
#inputs = tf.placeholder(tf.float32, shape=(1,4,3))
#expand = make_expand(3)
#print(expand.shape)

stack_params = {
        'layers': [(Locality0, {'d': 6}),
            Locality1,
            Locality1,
            Flatten
        ]
    }

network_params = {
        'batchsize': 1,
        'data_shape': (2, 3),
        'stack_params': stack_params
        }

import test_cfg
network_params2 = test_cfg.network_params

net = Network(network_params)

#s = Stack(inputs, stack_params)

#print("stack params")
#print(s.params)
with open("test_cfg.py", "w+") as f:
    f.write(net.generate_config())
#print(s.generate_config())

inputs = net.inputs
data = np.random.random(net.input_shape)
outs = net.outputs

print("Output shape:")
print(outs.get_shape().as_list())

init = tf.global_variables_initializer()

with tf.Session() as sess:
    sess.run(init)
    writer = tf.summary.FileWriter('testlogs', sess.graph)
    print(sess.run(outs, feed_dict={inputs:data}))
    writer.close()

