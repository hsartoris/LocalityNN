from Framework.Network import Hypervisor
from Framework.Network.tf_names import *
from Framework.Network.layers import *
from copy import deepcopy

stack_params = {
        'layers': [(Locality0, {'d': 6}),
            Locality1,
            (Flatten, {'activation': tanh})
        ]
    }

supervisor_params0 = {
        'stack_params': stack_params,
#        'load_from_ckpt': '/tmp/test2/supervisor0/15/1800.ckpt'
        }

stack_params1 = deepcopy(stack_params)
stack_params1['layers'][1] = Dumb1

supervisor_params = [
        supervisor_params0,
        { 
            'stack_params': stack_params1,
#            'load_from_ckpt': '/tmp/test2/supervisor1/18/1800.ckpt'
            }
        ]

hypervisor_params = {
        'batchsize': 10,
        'data_shape': (10, 5),
        'data_dir': 'tfrecords_test',
        'save_dir': '/tmp/test3',
        'train_item_count': 600,
        'valid_item_count': 200,
        'test_item_count': 200,
        'epochs': 40,
        'epochs_to_save': 10,
        'supervisor_params': supervisor_params,
        'batches_per_epoch': 60,
        'run_count': 20,
        }

hyp = Hypervisor(hypervisor_params)
print(hyp.generate_config())
hyp.run()
