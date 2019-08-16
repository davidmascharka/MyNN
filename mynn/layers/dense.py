from mygrad import matmul

from mynn.initializers.uniform import uniform
from mynn.initializers.constant import constant


class dense:
    """ A fully-connected layer.

    This class will perform a dense (fully-connected) linear operation on an (N, D)-shape
    input tensor with a (D, M)-shape weight tensor and a (M,)-shape bias.
    """

    def __init__(
            self,
            input_size,
            output_size,
            *,
            weight_initializer=uniform,
            bias_initializer=constant,
            weight_kwargs=None,
            bias_kwargs=None,
            bias=True,
    ):
        """ Initialize a dense layer.

        Parameters
        ----------
        input_size : int
            The number of features for each input datum.

        output_size : int
            The number of output units (neurons).

        weight_initializer : Callable, optional (default=initializers.uniform)
            The function to use to initialize the weight matrix.

        bias_initializer : Callable, optional (default=initializers.constant)
            The function to use to initialize the bias vector.

        weight_kwargs : Optional[dictionary]
            The keyword arguments to pass to the weight initialization function.

        bias_kwargs : Optional[dictionary]
            The keyword arguments to pass to the bias initialization function.

        bias : bool, optional (default=True)
            If `False` no bias parameter is initialized for the dense layer.
        """
        weight_kwargs = weight_kwargs if weight_kwargs is not None else {}
        bias_kwargs = bias_kwargs if bias_kwargs is not None else {}

        self.weight = weight_initializer(input_size, output_size, **weight_kwargs)
        self.bias = None
        if bias:
            self.bias = bias_initializer(1, output_size, **bias_kwargs)
            self.bias.data = self.bias.data.astype(self.weight.dtype)

    def __call__(self, x):
        """ Perform the forward-pass of the densely-connected layer over `x`.

        Parameters
        ----------
        x : Union[numpy.ndarray, mygrad.Tensor], shape=(N, D)
            The input to pass through the layer.

        Returns
        -------
        mygrad.Tensor
            The result of applying the dense layer wx + b.
        """
        out = matmul(x, self.weight)
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
