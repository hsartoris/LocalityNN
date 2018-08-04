Framework Implementation
===
Here, I provide an overview of the implementation of the framework as created so 
far. The codebase for the framework incorporates type hints to allow for 
checking of implementation validity.

For more specific implementation details, see the code comments.

common
---
As of the moment, `common` contains one class that both simulator and generator 
classes derive from: `Parameterizable`. This class provides parameter setting and 
getting, as well as the abstract class method `_get_default_params`, which is 
overridden on individual modules.

Generator
===
Several components make up the generator framework, all derived from 
`Parameterizable`.

`AbstractGenerator`
---
This class provides the base for `Generator` and modules. It passes off 
parameter management to `Parameterizable` and instantiates generator-specific 
variables, such as node count and matrix storage. Complete functions for getting 
and replacing the currently stored matrix are provided, as well as an abstract 
function for matrix generation, which must be overridden by modules.

`Generator`
---
This class extends `AbstractGenerator` and acts as a proxy for generator 
modules.  It takes such a module as a constructor argument and instantiates it 
with the other arguments passed.  All  function calls are passed to the 
instantiated module.  Thus all external code need only interact with `Generator` 
objects and only functions intended to be public are exposed.

Generator modules
---
Generator modules extend `AbstractGenerator` and must implement two private 
methods, `_get_default_params` and `_generate_structure`. Helper functions may 
be used, but only these two are accessed by the module's superclass functions, 
and thus by `Generator`. 

Simulator
===
As with the generator, simulator classes derive from `Parameterizable`.

`AbstractSimulator`
---
This class serves a purpose parallel to that of `AbstractGenerator`. After 
parameter instantiation by `Parameterizable`, the constructor stores the passed 
matrix and calls `_setup`. A complete implementation of `n_steps` is provided, 
depending on `step`, which must be overriden by modules along with `_setup`.  
`set_params`, derived from `Parameterizable`, is overridden to add a call to 
`_setup` after each parameter change.

`Simulator`
---
This class extends `AbstractSimulator` to the same end as `Generator`. 

Simulator modules
---
Simulator modules extend `AbstractSimulator` and must implement 
`_get_default_params` as well as `step`, which should return a vector 
representing the state of the network at the next timestep. As of right now, 
modules are responsible for storing the resulting vector as the current state.

Modules may optionally implement `_setup` if they require unique preparation 
operations to occur before simulation.
