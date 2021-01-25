import numpy as np

from mygrad.nnet.layers.conv import conv_nd

from mynn.initializers.uniform import uniform
from mynn.initializers.constant import constant

class conv:
    ''' A convolutional layer.

    This class will perform an n-dimensional convolution on an (N, K, ...)-shape input Tensor
    with a (D, K, ...,)-shape weight Tensor, then add a (D,)-shape bias vector to the result.
    '''
    def __init__(self, input_size, output_size, *filter_dims, stride=1, padding=0,
                 weight_initializer=uniform, bias_initializer=constant, weight_kwargs=None,
                 bias_kwargs=None):
        ''' Initialize a conv layer.

        Parameters
        ----------
        input_size : int
            The number of feature channels (depth) for each input datum.

        output_size : int
            The number of feature channels (depth) for the output.

        filter_dims : Sequence[int]
            The dimensions of the convolutional filters.

        stride : int, optional (default=1)
            The stride at which to move across the input.

        padding : int, optional (default=0)
            The amount of zero-padding to add before performing the forward-pass.

        weight_initializer : Callable, optional (default=initializers.uniform)
            The function to use to initialize the weight tensor.

        bias_initializer : Callable, optional (default=initializers.constant)
            The function to use to initialize the bias vector.

        weight_kwargs : Optional[dictionary]
            The keyword arguments to pass to the weight initialization function.

        bias_kwargs : Optional[dictionary]
            The keyword arguments to pass to the bias initialization function.
        '''
        if np.ndim(filter_dims) > 1:      # if the user passes in a Sequence
            filter_dims = filter_dims[0]  # unpack it from the outer Tuple

        weight_kwargs = weight_kwargs if weight_kwargs is not None else {}
        bias_kwargs = bias_kwargs if bias_kwargs is not None else {}

        self.weight = weight_initializer(output_size, input_size, *filter_dims, **weight_kwargs)
        self.bias = bias_initializer(output_size, **bias_kwargs).reshape(1, -1, *(1 for _ in range(len(filter_dims))))
        self.bias.data = self.bias.data.astype(self.weight.dtype)
        self.stride = stride
        self.padding = padding

    def __call__(self, x):
        ''' Perform the forward-pass of the n-dimensional convolutional layer over `x`.

        Parameters
        ----------
        x : Union[numpy.ndarray, mygrad.Tensor], shape=(N, K, ...)
            The data over which to perform convolution.

        Returns
        -------
        mygrad.Tensor
            The result of convolving `x` with this layer's `weight`, then adding its `bias`.
        '''
        return conv_nd(x, self.weight, stride=self.stride, padding=self.padding) + self.bias

    @property
    def parameters(self):
        ''' Access the parameters of the layer.

        Returns
        -------
        Tuple[mygrad.Tensor, mygrad.Tensor]
            The weight and bias of this layer.
        '''
        return self.weight, self.bias
