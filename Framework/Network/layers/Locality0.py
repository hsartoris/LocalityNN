from ..AbstractLayer import AbstractLayer
import tensorflow as tf
import numpy as np

class Locality0(AbstractLayer):
    """Modular implementation of first locality-based layer.

    Expected inputs are matrices of shape (timesteps x num_neurons), from which 
    the layer will derive most of its dimensionality.

    Parameters:
        d: int - chosen output depth. Defaults to timesteps/2
        batchsize: int - number of samples per batch
        stddev_w: float - standard deviation of weight matrices
        stddev_b: float - standard deviation of bias matrices
        weight_initializer: tf.keras.initializers.Initializer
        bias_initializer: tf.keras.initializers.Initializer

    """
    # TODO: add explanation of what the hell it actually does.

    # TODO: implement no-batch model via 1 batchsize and reshaping at end

    def _init_tiles(self) -> None:
        """Creates two constant matrices used in the functioning of this layer.
        
        `expand` appears as such for n == 3:
            | 1 1 1 0 0 0 0 0 0 |
            | 0 0 0 1 1 1 0 0 0 |
            | 0 0 0 0 0 0 1 1 1 |

        `tile` appears as such for n == 3:
            | 1 0 0 1 0 0 1 0 0 |
            | 0 1 0 0 1 0 0 1 0 |
            | 0 0 1 0 0 1 0 0 1 |
        """

        # TODO: abstract this into a general function to avoid duplication
        n: int = self.n

        expand: np.ndarray = np.array([[1]*n + [0]*n*(n-1)])
        for i in range(1, n):
            expand = np.append(expand, [[0]*n*i + [1]*n + [0]*n*(n-1-i)], 0)
        self.expand: tf.constant = tf.constant(expand, self.dtype)

        # TODO: `tile` may not be necessary; does tf.tile work for all cases?
        tile: np.ndarray = np.array([([1] + [0]*(n-1))*n])
        for i in range(1, n):
            tile = np.append(tile, [([0]*i + [1] + [0]*(n-1-i))*n], 0)
        self.tile: tf.constant = tf.constant(tile, self.dtype)


    def _setup(self) -> None:
        """Uses input dimensions to instantiate weight and bias matrices, as 
        well as required tiling matrices.
        """
        self.tsteps: int = self.input_shape[1]
        self.n: int = self.input_shape[2]
        assert(not self.tsteps == 0)
        assert(not self.n == 0)

        self._init_tiles()

        if self.params['d'] is None:
            self.params['d'] = int(self.tsteps/2)
    
