from mygrad import Tensor

from mynn.activations.leaky_relu import leaky_relu

class prelu:
    ''' A parametric rectified linear unit.

    This class maintains the learned slope parameter of a PReLU unit, which is a leaky ReLU with
    learned slope in the negative region.

    Parameters
    ----------
    slope : Real, optional (default=0.1)
        The initial value to use for the slope.
    '''
    def __init__(self, slope=0.1):
        self.slope = Tensor(slope)

    def __call__(self, x):
        return leaky_relu(x, self.slope)

    @property
    def parameters(self):
        ''' Access the parameters of the layer.

        Returns
        -------
        Tuple[mygrad.Tensor]
            The slope of the PReLU unit.
        '''
        return (self.slope,)
