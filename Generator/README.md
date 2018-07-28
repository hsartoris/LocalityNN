Usage
------------------------------------------------------------------------------------
To use a Generator module, import the Generator wrapper as well as the module:

```python
from Generator import Generator
from Generator.modules import <module>
```

Then create a Generator object with the module and node count as arguments:

```python
gen = Generator(<module>, <number>)
```
For an Erdos-Renyi generator on 5 nodes, this appears as follows:

```python
from Generator import Generator
from Generator.modules import Erdos_Renyi

gen = Generator(Erdos_Renyi, 5)
```

See `Template.py` for method documentation.

Creating Modules
------------------------------------------------------------------------------------
Generator modules are classes extending `Template`. The only methods that need 
to be overridden are `_get_default_params`, a class method, and 
`_generate_structure`.  Storing the resulting matrix is handled by `Template`.

Please use type hints.

Once your module is ready, place it in `Generator/modules/` and modify 
`Generator/modules/__init__.py` to include a line importing your module class:

```python
from .<filename> import <classname>
```
