from abc import ABC, abstractmethod
import tensorflow as tf
from typing import List, Callable, Dict, Tuple

class AbstractLayer(ABC):
    """Abstract class implementing parameterization for layers and stacks.

    """

    def __init__(self, inputs: tf.Tensor, params: Dict[str, any] = None,
            parent_params: Dict[str, any] = None):
        self.inputs: tf.Tensor = inputs

        # load default params for class and check that defaults/types contain 
        # same params
        self._load_default_params(self._import_default_params())

        self.params: Dict[str, any] = self.param_defaults.copy()

        # if parent params are provided, override defaults
        if parent_params is not None:
            self._override_params(parent_params, strict = False)

        # if layer-specific params are provided, override parent and defaults
        if params is not None:
            self._override_params(params)

        self._check_params()
        
        self._setup()
        self.outputs: tf.Tensor = self._compute_layer_ops()

    @abstractmethod
    def _compute_layer_ops(self) -> tf.Tensor:
        """Actual layer computations performed here.
        """


    @abstractmethod
    def _setup(self):
        """Operations for building the layer should be built here.
        """

        
    def _check_params(self) -> None:
        # check that requirements are satisfied
        for key in self.requirements:
            if self.params[key] is None:
                raise Exception("Param " + key +
                    " is required but is not provided.")

        # check that all types match
        for key in self.params:
            # since requirements are checked, Nonetype params can be skipped
            if self.params[key] is None: continue

            if not isinstance(self.params[key], self.param_types[key]):
                raise Exception("Param " + key + " has type " +
                        str(type(self.params[key])) + " but requires type " +
                        str(self.param_types[key]))

        
    def _override_params(self, new_params: Dict[str, any],
            strict: bool = True) -> None:

        for key in new_params:
            if key not in self.params:
                if strict:
                    raise Exception("Provided parameter " + key +
                        " is not supported on this module.")
                else: continue
            self.params[key] = new_params[key]

    @classmethod
    @abstractmethod
    def _import_default_params(cls) -> object:
        """Class method that should import default params for a given module and 
        return them.

        Object returned will be a module containing params_default, params_type, 
        and requirements dicts and lists.
        
        Must be overridden on module.
        """

    def _load_default_params(self, conf_file: object) -> None:
        """Pulls out actual items from imported config object.
        """
        # TODO: this is bad code

        # make sure all fields are present and typed appropriately
        if not hasattr(conf_file, "param_defaults"):
            # TODO: better exception
            raise Exception("Required attribute param_defaults not found")
        self.param_defaults: Dict[str, any] = conf_file.param_defaults

        if not hasattr(conf_file, "param_types"):
            raise Exception("Required attribute param_types not found")
        self.param_types: Dict[str, type] = conf_file.param_types

        if not set(self.param_defaults) == set(self.param_types):
            raise Exception("Default param value names and type names mismatch")

        if not hasattr(conf_file, "requirements"):
            raise Exception("required attribute requirements not found")
        self.requirements: List[str] = conf_file.requirements
