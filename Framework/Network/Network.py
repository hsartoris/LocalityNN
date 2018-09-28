from .NetworkModule import NetworkModule
from .Stack import Stack
import tensorflow as tf
from typing import List, Dict, Tuple
import os
from .util import parse_tfrecord

class Network(NetworkModule):
    """Builds, trains, and runs a complete network.
    
    Designed to be compatible with the TensorFlow Estimator library.

    Parameters:
        batchsize: int
        init_learn_rate: float
        optimizer
        loss_op
        prediction_activation

        stack_params: dict - dict defining the stack

    """
     
    def model_fn(self, features, labels, mode):
        """Function that calls Stack to build the network and defines ops 
        required for training and evaluation.

        Designed for use with tf.estimator.Estimator
        """
        # retrieve flat time series from features dict
        # why this is a dict I don't know
        inputs: tf.Tensor = features['time_series']

        # reshape inputs appropriately
        inputs = tf.reshape(inputs, list(self.input_shape))

        # send inputs through layer stack
        # outputs stored at stack.outputs
        self.stack: Stack = Stack(inputs,
                params = self.params['stack_params'],
                parent_params = self.params)

        predictions: Dict[str, any] = {
                # TODO: more stuff on predictions
                "probabilities": self.params['prediction_activation'](
                    self.stack.outputs, name="prediction_activation")
            }

        if mode == tf.estimator.ModeKeys.PREDICT:
            # prediction mode, duh
            return tf.estimator.EstimatorSpec(mode=mode, 
                    predictions=predictions)

        # loss calculation
        print("labels:", labels.get_shape().as_list())
        print("outs:", self.stack.outputs.get_shape().as_list())
        loss = self.params['loss_op'](labels = labels,
                predictions = self.stack.outputs)

        # training op
        if mode == tf.estimator.ModeKeys.TRAIN:
            optimizer = self.params['optimizer'](
                    learning_rate = self.params['init_learn_rate'])
            train_op = optimizer.minimize(
                    loss = loss,
                    global_step = tf.train.get_global_step())
            return tf.estimator.EstimatorSpec(mode = mode, loss = loss,
                    train_op = train_op)

        # EVAL mode stuff
        eval_metric_ops: Dict[str, any] = {
                "accuracy": tf.metrics.accuracy(
                    labels = labels,
                    # This is... sketchy
                    # on the other hand accuracy is a dumb metric so who cares
                    predictions = predictions['probabilities'])
                }
        if mode == tf.estimator.ModeKeys.EVAL:
            return tf.estimator.EstimatorSpec(
                    mode = mode, loss = loss, eval_metric_ops = eval_metric_ops)

        # unreachable
        assert(False)


    def _setup(self) -> None:
        self.data_shape: Tuple[int, int] = self.params['data_shape']

        self.input_shape: Tuple[int, int, int] = (self.params['batchsize'],
                self.data_shape[0], self.data_shape[1])

    def generate_config(self, indent_level: int = 0) -> str:
        if not hasattr(self, "stack"):
            return("""Due to restrictions imposed in part by the TensorFlow 
            Estimator framework and in part by decisions made in this framework, 
            Network cannot generate a config until model_fn has been called.""")

        conf: str = "from Framework.Network.layers import *\n"
        conf += "from Framework.Network.tf_names import *\n\n"
        conf += "network_params = {\n"
        idl: int = 1
        sp: int = idl * 4
        conf += self._generate_config(idl, skip = "stack_params")

        conf += " "*sp + "'stack_params': {\n"
        conf += self.stack.generate_config(idl+1)
        conf += " "*sp + "}\n"
        conf += "}\n"
        return conf


    @classmethod
    def _import_default_params(cls) -> object:
        from .conf import network
        return network

