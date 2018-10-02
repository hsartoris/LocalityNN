from Framework import Generator, Simulator
from Framework.Generator.modules import Erdos_Renyi
from Framework.Simulator.modules import matmul
from Framework.Network.util import Log as log
from Framework.Network.util import make_tfrecords
from Framework.common.Prettify import pretty
import numpy as np

p = pretty()

log.basicConfig(2)

gen = Generator(Erdos_Renyi, 5, params={'p': .5})
print(gen.get_structure())
sim = Simulator(matmul, gen.get_structure())
print(sim.n_steps(10).flatten())

# 5 neurons
# 10 timesteps
# labels flattened shape: (25)
# entries flattened shape: (50)

data = []
for i in range(1000):
    p.arrow(i, 1000)
    data.append(sim.n_steps(10))

label = gen.get_structure()

make_tfrecords(data, label, "tfrecords_test")
