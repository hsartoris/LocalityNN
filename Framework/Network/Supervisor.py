import numpy as np
from .NetworkModule import NetworkModule
from .Stack import Stack
import tensorflow as tf
from typing import List, Dict, Tuple
import os, sys
from .util import Log as log
from .util.dataset import load_dataset

class Supervisor(NetworkModule):
    TRAIN: str = "train"
    VALID: str = "validation"
    TEST: str = "testing"

    """Builds, trains, and runs a complete network.
    
    Pa:
        batchsize: int
        init_learn_rate: float
        optimizer
        loss_op
        prediction_activation

        stack_params: dict - dict defining the stack

        data_dir: str - location containing TFRecords
        save_dir: str - location for saving data

    """

    def save(self, global_step: int) -> None:
        self.saver.save(self.sess,
                os.path.join(self.save_dir_abs, str(global_step) + ".ckpt"))

    def predict(self) -> Tuple[np.ndarray, np.ndarray]:
        """Gets prediction and label for next item in current dataset.
        """
        preds, labels = self.sess.run([self.prediction, self.labels])
        return preds[0].reshape((5,5)), labels[0].reshape((5,5))

    def train(self) -> None:
        """Exectutes single training step.
        """
        self.sess.run(self.train_op)

    def validate(self, batches_per_epoch: int) -> None:
        """Reinitializes mean loss tracker as well as dataset; iterates through 
        validation dataset to get mean loss.
        """
        # replace iterator with validation data iterator
        self.sess.run(self.valid_init_op)
        for _ in range(batches_per_epoch):
            self.sess.run(self.mean_loss_op)
        # go back to training dataset
        self.sess.run(self.train_init_op)


    def summarize_epoch_loss(self, steps: int, global_step: int,
            mode: str = None) -> float:
        """Reinitializes iterator and runs through dataset, summarizing and 
        returning average loss."""
        if mode is None:
            mode = self.TRAIN

        log.info("Calculating epoch loss for", mode, "set")

        if mode == self.TRAIN:
            self.sess.run(self.train_init_op)
        elif mode == self.VALID:
            self.sess.run(self.valid_init_op)
        elif mode == self.TEST:
            self.sess.run(self.test_init_op)
        else:
            raise Exception("Bad mode: " + mode)

        for _ in range(steps):
            self.sess.run(self.mean_loss_op)

        mean_loss = self.sess.run(self.mean_loss)
        summary = self.sess.run(self.mean_loss_sum)


        if mode == self.TRAIN:
            self.train_writer.add_summary(summary, global_step)
        elif mode == self.VALID:
            self.valid_writer.add_summary(summary, global_step)
        elif mode == self.TEST:
            self.test_writer.add_summary(summary, global_step)

        self.sess.run(tf.local_variables_initializer())
        return mean_loss


    def _setup(self) -> None:
        # validate dirs and get absolute paths
        data_dir_abs, self.save_dir_abs = self.validate_dirs()

        # store batchsize for convenience
        self.batchsize: int = self.params['batchsize']

        # load datasets (requires self.batchsize)
        self.load_data(data_dir_abs)
        
        # make iterator
        log.debug("Creating dataset iterator...")
        self.iterator = tf.data.Iterator.from_structure(
                self.train_set.output_types,
                self.train_set.output_shapes)

        self.train_init_op = self.iterator.make_initializer(self.train_set)
        self.valid_init_op = self.iterator.make_initializer(self.valid_set)
        self.test_init_op = self.iterator.make_initializer(self.test_set)
        log.debug("...complete")

        self.inputs, self.labels = self.iterator.get_next()

        # store data_shape for convenience
        self.data_shape: Tuple[int, int] = self.params['data_shape']
        log.debug("Data shape:", self.data_shape)

        # store input_shape for convenience
        self.input_shape: Tuple[int, int, int] = (self.batchsize,
                self.data_shape[0], self.data_shape[1])
        log.debug("Batched input shape:", self.input_shape)
        
        # build graph (requires input_shape)
        self.build_graph()

        # at this point, prediction, loss, and train ops are defined

        # set up saver
        self.saver = tf.train.Saver()

        # set up summary ops

        self.mean_loss, self.mean_loss_op = tf.metrics.mean(self.loss)
        self.mean_loss_sum = tf.summary.scalar("loss", self.mean_loss)

        self.merged_sums = tf.summary.merge_all()


        # initialize variables and start session

        self.sess = tf.Session()
        tf.train.get_or_create_global_step()

        if self.params['load_from_ckpt'] is None:
            self.sess.run(tf.global_variables_initializer())
        else:
            # restoring from saved checkpoint
            log.info("Restoring from checkpoint", self.params['load_from_ckpt'])
            self.saver.restore(self.sess, self.params['load_from_ckpt'])

        self.sess.run(self.train_init_op)
        self.sess.run(tf.local_variables_initializer())
        self.train_writer = tf.summary.FileWriter(
                os.path.join(self.save_dir_abs, self.TRAIN), self.sess.graph)
        self.valid_writer = tf.summary.FileWriter(
                os.path.join(self.save_dir_abs, self.VALID))
        self.test_writer = tf.summary.FileWriter(
                os.path.join(self.save_dir_abs, self.TEST))

        # save initial state
        self.saver.save(self.sess, os.path.join(self.save_dir_abs, "init.ckpt"))


    def build_graph(self):
        log.info("Building graph")
        # reshape inputs appropriately
        self.inputs = tf.reshape(self.inputs, list(self.input_shape))

        # send inputs through layer stack
        # outputs stored at stack.outputs
        self.stack: Stack = Stack(self.inputs,
                params = self.params['stack_params'],
                parent_params = self.params)
        log.info("Completed Stack")

        log.info("Building ops")
        self.prediction = self.stack.outputs

        self.loss = self.params['loss_op'](labels = self.labels,
                predictions = self.stack.outputs)

        self.optimizer = self.params['optimizer'](
                learning_rate = self.params['init_learn_rate'])

        self.train_op = self.optimizer.minimize(
                loss = self.loss,
                global_step = tf.train.get_global_step())
        log.info("Ops complete")


    def validate_dirs(self) -> Tuple[str, str]:
        # check data dir
        data_dir_abs: str = os.path.join(os.getcwd(), self.params['data_dir'])
        if not os.path.isdir(data_dir_abs):
            log.critical("Data directory:", data_dir_abs, ", missing!")
            sys.exit()

        # check save dir
        save_dir_abs: str = os.path.join(os.getcwd(), self.params['save_dir'])
        if not os.path.isdir(save_dir_abs):
            log.info("Save directory (" + save_dir_abs +
                    ") not found; creating")
            os.makedirs(save_dir_abs)
        else:
            # directory exists; we're not gonna muck around with overwriting
            log.critical("Save directory (" + save_dir_abs +
                    ") already exists; aborting")
            sys.exit()

        return data_dir_abs, save_dir_abs


    def load_data(self, data_dir_abs):
        if self.params['shuffle_buffer_size'] is None:
            log.info("Setting shuffle buffer size to batchsize*2")
            self.params['shuffle_buffer_size'] = self.batchsize * 2

        log.info("Loading datasets from", data_dir_abs)

        load_dataset_wrapper = lambda data_dir, name, repeat: load_dataset(
                data_dir,
                name,
                shuffle_buffer_size = self.params['shuffle_buffer_size'],
                batchsize = self.batchsize,
                prefetch_buffer = self.params['prefetch_buffer'],
                num_parallel = self.params['dataset_num_parallel'],
                repeat = repeat)
        
        log.info("Train set regex:",
                os.path.join(data_dir_abs, self.TRAIN + "*.tfrecords"))
        # load training set with repetition on
        self.train_set = load_dataset_wrapper(data_dir_abs, self.TRAIN, True)
        log.info("Training dataset loaded")

        log.info("Validation set regex:",
                os.path.join(data_dir_abs, self.VALID + "*.tfrecords"))
        # load validation set with repetition on
        self.valid_set = load_dataset_wrapper(data_dir_abs, self.VALID, True)
        log.info("Validation set loaded")

        log.info("Testing set regex:",
                os.path.join(data_dir_abs, self.TEST + "*.tfrecords"))
        # load testing set with repetition off
        self.test_set = load_dataset_wrapper(data_dir_abs, self.TEST, False)
        log.info("Testing set loaded")
        

    def generate_config(self, indent_level: int = 0) -> str:
        idl: int = indent_level
        sp: int = idl * 4
        conf: str = self._generate_config(idl, skip = "stack_params")

        conf += " "*sp + "'stack_params': {\n"
        conf += self.stack.generate_config(idl+1)
        conf += " "*sp + "}\n"
        return conf


    @classmethod
    def _import_default_params(cls) -> object:
        from .conf import supervisor
        return supervisor

