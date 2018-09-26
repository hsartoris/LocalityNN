import tensorflow as tf
import numpy as np
from Framework.Network.Stack import Stack
from Framework.Network.layers import Locality0, Locality1, Flatten

d = 3
#inputs = tf.placeholder(tf.float32, shape=(1,4,3))
#expand = make_expand(3)
#print(expand.shape)

stack_params = {
        'batchsize': 1,
        'input_dims': (1,1,1),
        'layers': [(Locality0, {'d': 3}),
            Locality1,
            Locality1,
            Flatten
        ]
    }

inputs = tf.placeholder(tf.float32, (1,1,1))

s = Stack(inputs, stack_params)

print("stack params")
print(s.params)
#with open("test_cfg.py", "w+") as f:
#    f.write(s.generate_config())
print(s.generate_config())

data = np.array([[[1]]])

print(s.outputs.get_shape().as_list())

init = tf.global_variables_initializer()

with tf.Session() as sess:
    sess.run(init)
    writer = tf.summary.FileWriter('testlogs', sess.graph)
    print(sess.run(s.outputs, feed_dict={inputs:data}))
    writer.close()

