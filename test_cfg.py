from Framework.Network.layers import *
from Framework.Network.tf_names import *

hypervisor_params = {
    'batchsize': 10,
    'data_shape': (10, 5),
    'data_dir': 'tfrecords_test',
    'save_dir': '/tmp/test2',
    'train_item_count': 600,
    'valid_item_count': 200,
    'test_item_count': 200,
    'batches_per_epoch': 60,
    'epochs': 20,
    'epochs_to_save': 10,
    'supervisor_params': {
        'batchsize': 10,
        'data_shape': (10, 5),
        'init_learn_rate': 0.0005,
        'optimizer': Adam,
        'loss_op': modified_mse,
        'data_dir': 'tfrecords_test',
        'save_dir': '/tmp/test2',
        'shuffle_buffer_size': 20,
        'prefetch_buffer': 1,
        'dataset_num_parallel': 2,
        'load_from_ckpt': None,
        'stack_params': {
            'batchsize': 10,
            'debug': True,
            'layers': [
                (Locality0, 'Locality0',
                    {
                        'd': 6,
                        'input_shape': (10, 10, 5),
                        'batchsize': 10,
                        'stddev_w': 0.25,
                        'stddev_b': 0.25,
                        'weight_initializer': random_normal,
                        'bias_initializer': random_normal,
                        'activation': relu,
                        'dtype': tf.float32,
                    }
                ),
                (Locality1, 'Locality1',
                    {
                        'input_shape': (10, 6, 25),
                        'batchsize': 10,
                        'activation': relu,
                        'dtype': tf.float32,
                        'stddev_w': 0.25,
                        'initializer_w': random_normal,
                        'stddev_b': 0.25,
                        'initializer_b': random_normal,
                        'stddev_w0': None,
                        'stddev_w1': None,
                        'stddev_w2': None,
                        'initializer_w0': None,
                        'initializer_w1': None,
                        'initializer_w2': None,
                    }
                ),
                (Flatten, 'Flatten',
                    {
                        'input_shape': (10, 6, 25),
                        'stddev_w': 1.0,
                        'initializer_w': random_normal,
                        'activation': tanh,
                        'dtype': tf.float32,
                    }
                ),
            ]
        }
    }
}
