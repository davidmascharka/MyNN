import numpy as np
from mygrad import Tensor
from mygrad.operations import Operation

class LeakyReLU(Operation):
    ''' Returns the leaky rectified linear activation elementwise along x. The leaky ReLU is given
    by max(x, 0) + slope*min(x, 0)
    '''
    def __call__(self, x, slope):
        '''
        Parameters
        ----------
        x : mygrad.Tensor
            Input data.

        slope : Real
            The slope of the negative activation.

        Returns
        -------
        numpy.ndarray
            The leaky-rectified `x` (elementwise max(x, 0) + slope*min(x, 0)).
        '''
        self.variables = (x,)
        return np.maximum(x.data, 0) + slope*np.minimum(x.data, 0)

    def backward_var(self, grad, index, **kwargs):
        x = self.variables[index]
        x.backward(grad, **kwargs)

def leaky_relu(x, slope):
    ''' Returns the leaky rectified linear activation elementwise along x. The leaky ReLU is given
    by max(x, 0) + slope*min(x, 0).

    Parameters
    ----------
    x : mygrad.Tensor
        Input data.

    slope : Real
        The slop of the negative activation.

    Returns
    -------
    numpy.ndarray
        The rectified `x` (elementwise max(x, 0)).
    '''
    return Tensor._op(LeakyReLU, x, op_args=(slope,))
