import numpy as np
from mygrad import Tensor
from mygrad.operation_base import Operation

class Sigmoid(Operation):
    ''' Returns f(x) = 1 / (1 + exp(-x))
    '''
    def __call__(self, x):
        '''Sigmoid activation
        Parameters
        ----------
        x : mygrad.Tensor
            Input data.
        
        Returns
        -------
        numpy.ndarray
            elementwise activation f(x) = 1 / (1 + exp(-x))
        '''
        self.variables = (x,)
        self.sigmoid = np.reciprocal(np.exp(-1 * x.data) + 1)
        return self.sigmoid

    def backward_var(self, grad, index, **kwargs):
        self.variables[index].backward(grad * self.sigmoid * (1.0 - self.sigmoid), **kwargs)

def sigmoid(x, min_val = -1, max_val = 1):
    ''' returns the sigmoid activation f(x) = 1 / (1 + exp(-x))

    Parameters
    ----------
    x : mygrad.Tensor
        Input data.

    Returns
    -------
    numpy.ndarray
        elementwise activation f(x) = 1 / (1 + exp(-x))
    '''
    return Tensor._op(Sigmoid, x)
