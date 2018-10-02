from Framework.Network import Hypervisor
from Framework.Network.tf_names import *
from Framework.Network.layers import *

stack_params = {
        'layers': [(Locality0, {'d': 6}),
            Locality1,
            (Flatten, {'activation': tanh})
        ]
    }

supervisor_params = {
        'stack_params': stack_params,
        }

hypervisor_params = {
        'batchsize': 10,
        'data_shape': (10, 5),
        'data_dir': 'tfrecords_test',
        'save_dir': '/tmp/test2',
        'train_item_count': 600,
        'valid_item_count': 200,
        'test_item_count': 200,
        'epochs': 20,
        'epochs_to_save': 10,
        'supervisor_params': supervisor_params,
        'batches_per_epoch': 60,
        }

hyp = Hypervisor(hypervisor_params)
print(hyp.generate_config())
#hyp.run()
