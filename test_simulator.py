from Framework import Generator, Simulator
from Framework.Generator.modules import Erdos_Renyi
from Framework.Simulator.modules import matmul

gen = Generator(Erdos_Renyi, 5)
print(gen.get_structure())
print(gen.get_structure())
sim = Simulator(matmul, gen.get_structure())
print(sim.step())
print(sim.n_steps(5))
