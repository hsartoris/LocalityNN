import tensorflow as tf
import numpy as np
from Framework.Network.tf_names import *
from Framework.Network import Supervisor
from Framework.Network.Stack import Stack
from Framework.Network.layers import Locality0, Locality1, Flatten
import os


# this sort of parameter will not be hardcoded like this eventually
tfrecords_dir = "tfrecords_test"
batchsize = 10
batches_per_epoch = 60
epochs = 40
timesteps = 10
num_nodes = 5
shuffle_buffer_size = 500

train_steps = 5000


stack_params = {
        'layers': [(Locality0, {'d': 6}),
            Locality1,
            (Flatten, {'activation': tanh})
        ]
    }

network_params = {
        'batchsize': batchsize,
        'data_shape': (timesteps, num_nodes),
        'init_learn_rate': .0005,
        'stack_params': stack_params,
        'data_dir': 'tfrecords_test',
        'save_dir': '/tmp/test2',
        'load_from_ckpt': '/tmp/test/init.ckpt'
        }

net = Supervisor(network_params)

for i in range(epochs):
    print("epoch", i)
    net.validate(200)
    vloss = net.summarize_loss(i*batches_per_epoch, net.VALID)
    for _ in range(batches_per_epoch):
        net.train()
    loss = net.summarize_loss(i*batches_per_epoch)
    print("Training loss: ", loss)
    print("Validation loss: ", vloss)
    net.save(i*batches_per_epoch)

pred, label = net.predict()
print(pred)
print(label)
print(np.sum(pred))


#init = tf.global_variables_initializer()
#
#with tf.Session() as sess:
#    sess.run(init)
#    writer = tf.summary.FileWriter('testlogs', sess.graph)
#    print(sess.run(outs))
#    writer.close()

