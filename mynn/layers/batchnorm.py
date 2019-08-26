import numpy as np
import mygrad as mg

from mygrad.nnet.layers.batchnorm import batchnorm as tensor_batchnorm


class batchnorm:
    """ A batch normalization layer.

    This class will perform an n-dimensional batch normalization operation on an
    (N, D, ...)-shaped tensor scaled by γ of shape (D, ...) and shifted by β of shape (D, ...).
    """

    def __init__(self, input_channels, momentum=0.1):
        """ Initialize a batch normalization layer.

        Parameters
        ----------
        input_channels : int
            The number of channels of the data to be batch-normalized.

        momentum : float, optional (default=0.1)
            The momentum value used to maintain moving averages.
        """
        self.gamma = mg.ones((1, input_channels), dtype=np.float32)
        self.beta = mg.zeros((1, input_channels), dtype=np.float32)

        self.moving_mean = np.zeros((1, input_channels), dtype=np.float32)
        self.moving_variance = np.zeros((1, input_channels), dtype=np.float32)

        self.momentum = momentum

        self.input_channels = input_channels

    def __call__(self, x, test=False):
        """ Perform the forward-pass of n-dimensional batch normalization over axis 1 on `x`.

        Parameters
        ----------
        x : Union[numpy.ndarray, mygrad.Tensor], shape=(N, D, ...)
            The data to normalize.

        test : boolean, optional (default=False)
            Determines whether the layer is being used at training time. The mean and variance
            will be computed for the batch during training, while averaged batch statistics will
            be used at test time.
        """
        if test:
            keepdims_shape = tuple(1 if n != 1 else d for n, d in enumerate(x.shape))
            x = x - self.moving_mean.reshape(keepdims_shape)
            x /= np.sqrt(self.moving_variance.reshape(keepdims_shape) + 1e-08)
            return self.gamma * x + self.beta

        x_norm = tensor_batchnorm(x, gamma=self.gamma, beta=self.beta, eps=1e-08)

        batch_mean = x_norm.creator.mean
        batch_variance = x_norm.creator.var

        self.moving_mean *= 1 - self.momentum
        self.moving_mean += self.momentum * batch_mean
        self.moving_variance *= 1 - self.momentum
        self.moving_variance += self.momentum * batch_variance
        return x_norm

    @property
    def parameters(self):
        """ Access the learnable parameters of the layer.

        Returns
        -------
        Tuple[mygrad.Tensor, mygrad.Tensor]
            The gamma and beta values of this layer.
        """
        return (self.gamma, self.beta)
