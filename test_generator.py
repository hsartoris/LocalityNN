from Generator import Generator
from Generator.modules import Erdos_Renyi

gen = Generator(Erdos_Renyi, 5)
print(gen.get_structure())
print(gen.new_structure())
gen.set_params({'p': .5})
print(gen.new_structure())
