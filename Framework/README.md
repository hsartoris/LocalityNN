Framework Implementation
===
Here, I provide an overview of the implementation of the framework as created so 
far.

common
===
As of the moment, `common` contains one class that both simulator and generator 
classes derive from: `AbstractBase`. This class provides parameter setting and 
getting, as well as the abstract class method `_get_default_params`, which is 
overridden on individual modules.

Generator
===
Several components make up the generator framework, all derived from 
`AbstractBase`.

`AbstractGenerator`
---
This class provides the base for `Generator` and modules. It passes off 
parameter management to `AbstractBase` and instantiates generator-specific 
variables, such as node count and matrix. Complete functions for getting and 
replacing the currently stored matrix are provided, as well as an abstract 
function for matrix generation, which must be overridden by modules.

`Generator`
---
This class extends `AbstractGenerator` and acts as a proxy for generator 
modules.  It takes such a module as a constructor argument and instantiates it 
with the other arguments passed.  All  function calls are passed to the 
instantiated module.  Thus all external code need only interact with `Generator` 
objects and only functions intended to be public are exposed.

modules
---
Generator modules extend `AbstractGenerator` and must implement two private 
methods, `_get_default_params` and `_generate_structure`. Helper functions may 
be used, but only these two are accessed by the module's superclass functions, 
and thus by `Generator`. 
