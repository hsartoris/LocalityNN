from Framework.Network.Stack import Stack
from Framework.Network.layers import Locality0

d = 3
#inputs = tf.placeholder(tf.float32, shape=(1,4,3))
#expand = make_expand(3)
#print(expand.shape)

stack_params = {
        'batchsize': 1,
        'input_dims': (1,1,1),
        'layers': [
            (Locality0, 'test',
                {
                    'd': 3,
                    'input_shape': (1,1,1)
                }
            )
        ]
    }

s = Stack(stack_params)

print("stack params")
print(s.params)
print(s.generate_config())
with open("test_cfg.py", "w+") as f:
    f.write(s.generate_config())

