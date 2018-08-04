from ..common import Parameterizable
import tensorflow as tf
from typing import List, Callable, Dict, Tuple
from .util import JSONDecoder, encode_json

confdir = "conf/"

class AbstractLayer(Parameterizable):
    """Abstract class for wrapping layer ops into a full layer.

    Extends Parameterizable for parameter management.

    Child classes can optionally override _setup, and must override _layer_ops.

    Any and all calls to create variables in these methods should use 
    tf.get_variable, to maintain name scoping.

    `__init__` does not provide default values. Those are provided by the 
    wrapper class, Layer.
    
    """

    def __init__(self, inputs: tf.Tensor, params: Dict) -> None:
        # call superclass initializer with params as argument
        super(AbstractLayer, self).__init__(params)

        # record provided information. not everything will be used by every 
        # layer implementation, but all values should be generally useful
        self.inputs: tf.Tensor = inputs
        self.input_shape: List[int] = inputs.get_shape().as_list()
        self.dtype: type = inputs.dtype
        self.activation: Callable[[tf.Tensor, str], tf.Tensor] = \
                self.params['activation']

        self.batchsize: int = self.params['batchsize']

        self.name: str = self.params['name']
        
        # TODO: convert this to a logging message
        self.use_scope: bool = self.names is not None

        if not self.use_scope:
            print("Warning: this instance of " + type(self).__name__ +
                    " is running unscoped.")

        # allow module to retrieve globally scoped constants
        self._get_global_constants()

        # run local operations, with scope if name is provided
        if self.use_scope:
            with tf.variable_scope(self.name):
                self._setup()
                self.outputs: tf.Tensor = self._layer_ops()
        else:
            self._setup()
            self.outputs: tf.Tensor = self._layer_ops()

    @classmethod
    def _load_config(cls) -> None:
        # load configuration information without instantiating
        if not hasattr(cls, "defaults"):
            cls.json_decoder: JSONDecoder = JSONDecoder()
            with open(cls.confdir + cls.__name__ + ".json") as f:
                params: Dict[str, any] = cls.json_decoder.load_json(f)
            cls.defaults: Dict[str, any] = params['defaults']
            cls.types: Dict[str, type] = params['types']
            cls.requirements: List[str] = params['requirements']

    @classmethod
    def _get_default_params(cls) -> Dict[str, any]:
        """Load config file for module. Should obviate implementing this method 
        on individual modules.
        """
        cls._load_config()
        return cls.defaults

    def _get_global_constants(self) -> None:
        """Optional method to allow modules to retrieve global constants before 
        scoping is used. Must use tf.get_variable.
        """
        pass

    def _setup(self) -> None:
        """Any operations that should be run before _layer_ops. Optional.

        Override on subclasses.
        """
        pass


    def _layer_ops(self) -> tf.Tensor:
        """Responsible for actual layer computations.

        Must be overridden by child classes.
        """
        raise NotImplementedError

    
