import tensorflow as tf
import numpy as np
from Framework.Network.tf_names import *
from Framework.Network import Network
from Framework.Network.Stack import Stack
from Framework.Network.layers import Locality0, Locality1, Flatten
from Framework.Network.util.dataset import load_to_input_fn
import os

#os.environ['TF_CPP_MIN_LOG_LEVEL'] = '0'
tf.logging.set_verbosity(tf.logging.INFO)

# this sort of parameter will not be hardcoded like this eventually
tfrecords_dir = "tfrecords_test"
batchsize = 2
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
        'init_learn_rate': .001,
        'stack_params': stack_params
        }

net = Network(network_params)

est = tf.estimator.Estimator(model_fn = net.model_fn, model_dir="/tmp/model")

tensors_to_log = {"probabilities": "prediction_activation"}
logging_hook = tf.train.LoggingTensorHook(
        tensors = tensors_to_log,
        every_n_iter = 250)

est.train(input_fn = lambda: load_to_input_fn(
                                    data_dir = tfrecords_dir,
                                    name = "train",
                                    shuffle_buffer_size = shuffle_buffer_size,
                                    batchsize = batchsize),
        steps = train_steps,
        hooks = [logging_hook])

eval_results = est.evaluate(input_fn = lambda: load_to_input_fn(
                                    data_dir = tfrecords_dir,
                                    name = "testing",
                                    shuffle_buffer_size = shuffle_buffer_size,
                                    batchsize = batchsize,
                                    repeat = False))
print(eval_results)

#init = tf.global_variables_initializer()
#
#with tf.Session() as sess:
#    sess.run(init)
#    writer = tf.summary.FileWriter('testlogs', sess.graph)
#    print(sess.run(outs))
#    writer.close()

