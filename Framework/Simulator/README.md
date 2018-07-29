Usage
---
To use a Simulator module, import the Simulator wrapper as well as the module:

```python
from Framework import Simulator
from Framework.Simulator.modules import <module>
```

Then create a Simulator object with the module and chosen adjacency matrix as 
arguments:

```python
sim = Simulator(<module>, <matrix>)
```

For a matrix multiplication simulator on some matrix `mat`, this appears as 
follows:

```python
from Framework import simulator
from Framework.Simulator.modules import matmul

sim = Simulator(matmul, mat)
```

See `Simulator.py` for public method documentation.

Creating Modules
---
Simulator modules are classes extending `AbstractSimulator`. The methods that 
must be overriden are `_get_default_params` and `step`. Note that 
`_get_default_params` is a class method. `n_steps` relies on `step` and is 
provided by `AbstractSimulator`.

Additionally, `_setup` is called by `AbstractSimulator` at the end of 
`__init__`. By default it returns with no action. Modules can override it to 
perform unique setup operations.

Once the module is ready, place it in `Framework/Simulator/modules/` and modify 
`Framework/Simulator/modules/__init__.py` to include a line importing the module 
class:

```python
from .<filename> import <classname>
```

For the class `matmul`, located in `matmul.py`:

```python
from .matmul import matmul
```

TODO
---
1. Implement more modules
	* Izhikevich
