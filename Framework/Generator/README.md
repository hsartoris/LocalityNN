Usage
---
To use a Generator module, import the Generator wrapper as well as the module:

```python
from Framework import Generator
from Framework.Generator.modules import <module>
```

Then create a Generator object with the module and node count as arguments:

```python
gen = Generator(<module>, <number>)
```
For an Erdos-Renyi generator on 5 nodes, this appears as follows:

```python
from Framework import Generator
from Framework.Generator.modules import Erdos_Renyi

gen = Generator(Erdos_Renyi, 5)
```

See `Generator.py` for public method documentation.

Creating Modules
---
Generator modules are classes extending `AbstractGenerator`. The only methods 
that strictly need to be overridden are `_get_default_params`, a class method, 
and `_generate_structure`.  Storing the resulting matrix is handled by 
`AbstractGenerator`.  Return types for both methods are hinted in 
`AbsractGenerator`; please hint appropriately.

Once your module is ready, place it in `Framework/Generator/modules/` and modify 
`Framework/Generator/modules/__init__.py` to include a line importing your 
module class:

```python
from .<filename> import <classname>
```

For the class `Erdos_Renyi`, located in `erdos_renyi.py`:

```python
from .erdos_renyi import Erdos_Renyi
```

TODO
---
1. Implement more generators
	* Finish Barabasi-Albert
2. Delineate common parameters vs unique parameters; e.g. node count & 
   probability
	* perhaps separate dicts and methods
3. Optional capacity for parameter validation (within range or the like)
