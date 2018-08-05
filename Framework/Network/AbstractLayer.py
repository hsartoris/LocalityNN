from abc import ABC, abstractmethod
import tensorflow as tf
from typing import List, Callable, Dict, Tuple
from .util import JSONDecoder, encode_json

confdir = "conf/"

class AbstractLayer(ABC):
    """Abstract class for wrapping layer ops into a full layer.

    Does not use Parameterizable because Tensorflow graphs might as well be 
    immutable.

    Child classes can optionally override _setup, and must override _layer_ops.

    Any and all calls to create variables in these methods should use 
    tf.get_variable, to maintain name scoping.

    `__init__` does not provide default values. Those are provided by the 
    wrapper class, Layer.

    Before using a module, load its default configuration from 
    conf/<layername>.json, using util.JSONDecoder. Then pass the resulting 
    object to set_class_defaults. No values are stored in the code itself, and 
    skipping this step will result in failure to initialize the model later.

    TODO: just make this a method on the class
    
    """

    # TODO: this is hacky
    general_conf_name: str = "AbstractLayer"

    @classmethod
    def set_class_defaults(cls, conf: Dict[str, any]) -> None:
        """Allows managing class, likely NetworkBuilder, to pass in the defaults 
        for the class, which are stored in conf/<layer>.json

        This method does basic validation of the config as passed in.
        """
        # TODO: ABSTRACT INTO CONF HANDLER OBJECT.

        # to summarize, if we get here, the following is true:
        #   1. the sections defined in the config are exactly the required 3
        #   2. the items in the three sections are Dict, Dict, List
        #   3. the params defined in `defaults` and `types` are equivalent
        #   4. any required params are defined in `defaults` and `types`
        # now we just store the variables and we're done
        cls.default_params: Dict[str, any] = defaults
        cls.param_types: Dict[str, type] = types
        cls.reqd_params: List[str] = reqs


    def __init__(self, inputs: tf.Tensor, params: Dict) -> None:
        # compare provided params against defaults
        self._validate_params(params)

        # save hook to inputs and derive dimensionality from it
        self.inputs: tf.Tensor = inputs
        self.input_shape: List[int] = inputs.get_shape().as_list()
        self.activation: Callable[[tf.Tensor, str], tf.Tensor] = \
                self.params['activation']

        # TODO: probably pass in the type
        self.dtype: type = inputs.dtype

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
    def load_config(cls, genl_conf = general_conf_name) -> None:
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

    @abstractmethod
    def _get_global_constants(self) -> None:
        """Optional method to allow modules to retrieve global constants before 
        scoping is used. Must use tf.get_variable.
        """
        pass

    @abstractmethod
    def _setup(self) -> None:
        """Any operations that should be run before _layer_ops. Optional.

        Override on subclasses.
        """
        pass

    @abstractmethod
    def _layer_ops(self) -> tf.Tensor:
        """Responsible for actual layer computations.

        Must be overridden by child classes.
        """
