from typing import Callable, Dict, List, Tuple
import tensorflow as tf

params: type = Dict[str, Tuple[any, type, bool]]

activation: type = Callable[[tf.Tensor, str], tf.Tensor]
