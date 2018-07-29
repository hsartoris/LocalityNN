Project Info
---
The eventual goal of this project is to provide a framework for generating 
neural network simulation data from various generated graph structures and 
training artificial neural networks to deduce the original graph structure from 
the generated data.

Framework
---
The framework will eventually consist of three parts: generator, simulator, and 
ANN. Currently, the generator and simulator have complete foundations, and 
Erdos-Renyi generation and simple matrix multiplication modules are ready.

The generator and simulator foundations are designed to provide all of the 
functionality possible to abstract away from individual modules, as well as 
boilerplate code used by other code to set parameters and retrieve outputs. No 
matter what network generator, the functions used to interact with the generator 
are the same, provided by a proxy `Generator` class. The same is true of 
simulators.

Computation specific to different methods of generation and simulation is left 
up to module classes, which simply implement standard functions to communicate 
with their foundations.

This approach allows for simple implementation and use of new modules. More 
concretely, the Erdos-Renyi network generator module contains only two methods, 
with a total of seven lines of operating code between them.

For a more technical examination of the framework structure, see 
`Framework/README.md`. For usage details and module creation instructions, see 
the READMEs in `Framework/Generator/` and `Framework/Simulator/`

TODO
---
There's a lot of code reuse between the Generator and Simulator templates.  
Abstract the parts of functionality that are common, such as `get_params` etc, 
into a base template, and rename the Generator and Simulator templates to 
reflect individual status.

^ Pretty much done
