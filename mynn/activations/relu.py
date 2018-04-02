import numpy as np
from mygrad import Tensor
from mygrad.operations import Operation

class ReLU(Operation):
    ''' Returns the rectified linear activation max(x, 0) elementwise along x. '''
    def __call__(self, x):
        '''
        Parameters
        ----------
        x : mygrad.Tensor
            Input data.

        Returns
        -------
        numpy.ndarray
            The rectified `x` (elementwise max(x, 0)).
        '''
        self.variables = (x,)
        return np.maximum(x.data, 0)

    def backward_var(self, grad, index, **kwargs):
        x = self.variables[index]
        x.backward(grad * np.piecewise(x.data, [x.data <= 0, x.data > 0], [0, 1]), **kwargs)

def relu(x):
    ''' Returns the rectified linear activation max(x, 0) elementwise along x.

    Parameters
    ----------
    x : mygrad.Tensor
        Input data.

    Returns
    -------
    numpy.ndarray
        The rectified `x` (elementwise max(x, 0)).
    '''
    return Tensor._op(ReLU, x)
