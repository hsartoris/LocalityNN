from .NetworkModule import NetworkModule
from .Stack import Stack
import tensorflow as tf
from typing import List, Dict, Tuple
import os, sys
from .util import Log as log
from .util.dataset import load_dataset

class Supervisor(NetworkModule):
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


    def _setup(self) -> None:
        log.basicConfig(level=1)

        # validate dirs and get absolute paths
        data_dir_abs, save_dir_abs = self.validate_dirs()
        # make log directory
        os.mkdir(os.path.join(save_dir_abs, "logs"))
        log_path: str = os.path.join(save_dir_abs, "logs", "log")
        log.file_out(log_path)

        # store batchsize for convenience
        self.batchsize: int = self.params['batchsize']

        # load datasets (requires self.batchsize)
        self.load_data(data_dir_abs)
        
        # make iterator
        log.debug("Creating dataset iterator...")
        self.iterator = tf.data.Iterator.from_structure(
                self.train_set.output_types,
                self.train_set.output_shapes)

        train_init_op = self.iterator.make_initializer(self.train_set)
        valid_init_op = self.iterator.make_initializer(self.valid_set)
        test_init_op = self.iterator.make_initializer(self.test_set)
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
        train = "train"
        valid = "validation"
        test = "testing"

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
                os.path.join(data_dir_abs, train + "*.tfrecords"))
        # load training set with repetition on
        self.train_set = load_dataset_wrapper(data_dir_abs, train, True)
        log.info("Training dataset loaded")

        log.info("Validation set regex:",
                os.path.join(data_dir_abs, valid + "*.tfrecords"))
        # load validation set with repetition on
        self.valid_set = load_dataset_wrapper(data_dir_abs, valid, True)
        log.info("Validation set loaded")

        log.info("Testing set regex:",
                os.path.join(data_dir_abs, test + "*.tfrecords"))
        # load testing set with repetition off
        self.test_set = load_dataset_wrapper(data_dir_abs, test, False)
        log.info("Testing set loaded")
        

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
        from .conf import supervisor
        return supervisor

