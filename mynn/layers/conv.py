import numpy as np

from mygrad.nnet.layers.conv import conv_nd

from mynn.initializers.uniform import uniform
from mynn.initializers.constant import constant


class conv:
    """ A convolutional layer.

    This class will perform an n-dimensional convolution on an (N, K, ...)-shape input Tensor
    with a (D, K, ...,)-shape weight Tensor, then add a (D,)-shape bias vector to the result.
    """

    def __init__(
            self,
            input_size,
            output_size,
            *filter_dims,
            stride=1,
            padding=0,
            dilation=1,
            weight_initializer=uniform,
            bias_initializer=constant,
            weight_kwargs=None,
            bias_kwargs=None,
            bias=True,
    ):
        """ Initialize a conv layer.

        Parameters
        ----------
        input_size : int
            The number of feature channels (depth) for each input datum.

        output_size : int
            The number of feature channels (depth) for the output.

        filter_dims : Sequence[int]
            The dimensions of the convolutional filters.

        stride : Union[int, Tuple[int, ...]]
            (keyword-only argument) The step-size with which each
            filter is placed along the H and W axes during the
            convolution. The tuple indicates (stride-0, ...). If a
            single integer is provided, this stride is used for all
            convolved dimensions

        padding : Union[int, Tuple[int, ...]], optional (default=0)
            (keyword-only argument) The number of zeros to be padded
            to both ends of each convolved dimension, respectively.
            If a single integer is provided, this padding is used for
            all of the convolved axes

        dilation : Union[int, Tuple[int, ...]], optional (default=1)
            (keyword-only argument) The spacing used when placing kernel
            elements along the data. E.g. for a 1D convolution the ith
            placement of the kernel multiplied  against the dilated-window:
            `x[:, :, i*s:(i*s + w*d):d]`, where s is
            the stride, w is the kernel-size, and d is the dilation factor.

            If a single integer is provided, that dilation value is used for all
            of the convolved axes.

        weight_initializer : Callable, optional (default=initializers.uniform)
            The function to use to initialize the weight tensor.

        bias_initializer : Callable, optional (default=initializers.constant)
            The function to use to initialize the bias vector.

        weight_kwargs : Optional[dictionary]
            The keyword arguments to pass to the weight initialization function.

        bias_kwargs : Optional[dictionary]
            The keyword arguments to pass to the bias initialization function.

        bias : bool, optional (default=True)
            If `False`, no biar parameter is initialized for the convolutional layer.
        """
        if np.ndim(filter_dims) > 1:  # if the user passes in a Sequence
            filter_dims = filter_dims[0]  # unpack it from the outer Tuple

        weight_kwargs = weight_kwargs if weight_kwargs is not None else {}
        bias_kwargs = bias_kwargs if bias_kwargs is not None else {}

        self.weight = weight_initializer(
            output_size, input_size, *filter_dims, **weight_kwargs
        )
        self.bias = None
        if bias:
            self.bias = bias_initializer(output_size, **bias_kwargs)
            self.bias = self.bias.reshape(1, -1, *(1 for _ in range(len(filter_dims))))
            self.bias.data = self.bias.data.astype(self.weight.dtype)
        self.stride = stride
        self.padding = padding
        self.dilation = dilation

    def __call__(self, x):
        """ Perform the forward-pass of the n-dimensional convolutional layer over `x`.

        Parameters
        ----------
        x : Union[numpy.ndarray, mygrad.Tensor], shape=(N, K, ...)
            The data over which to perform convolution.

        Returns
        -------
        mygrad.Tensor
            The result of convolving `x` with this layer's `weight`, then adding its `bias`.
        """
        out = conv_nd(
            x,
            self.weight,
            stride=self.stride,
            padding=self.padding,
            dilation=self.dilation,
        )
        return out + self.bias if self.bias is not None else out

    @property
    def parameters(self):
        """ Access the parameters of the layer.

        Returns
        -------
        Tuple[mygrad.Tensor, mygrad.Tensor]
            The weight and bias of this layer.
        """
        return (self.weight, self.bias) if self.bias is not None else (self.weight,)
