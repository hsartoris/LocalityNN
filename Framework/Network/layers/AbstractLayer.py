import tensorflow as tf
from typing import Callable

class AbstractLayer(object):
    def __init__(self) -> None:
        self.activation: Callable

