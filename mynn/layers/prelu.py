from mygrad import Tensor

from mynn.activations.leaky_relu import leaky_relu


class prelu:
    """ A parametric rectified linear unit.

    This class maintains the learned slope parameter of a PReLU unit, which is a leaky ReLU with
    learned slope in the negative region.

    """

    def __init__(self, slope=0.1):
        """ Parameters
            ----------
            slope : Real, optional (default=0.1)
                The initial value to use for the slope."""
        self.slope = Tensor(slope)

    def __call__(self, x):
        """ Forward the input through the PReLU.

        Parameters
        ----------
        x : Union[numpy.ndarray, mygrad.Tensor]
            The data, to which to apply the PReLU.

        Returns
        -------
        mygrad.Tensor
            The result of applying the PReLU elementwise across `x`.
        """
        return leaky_relu(x, self.slope)

    @property
    def parameters(self):
        """ Access the parameters of the layer.

        Returns
        -------
        Tuple[mygrad.Tensor]
            The slope of the PReLU unit.
        """
        return (self.slope,)
