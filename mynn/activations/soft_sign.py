import numpy as np
from mygrad import Tensor
from mygrad.operation_base import Operation
from mygrad import abs

class SoftSign(Operation):
    ''' Returns the soft sign function x/(1 + |x|) elementwise along x '''
    def __call__(self, x):
        '''
        Parameters
        ----------
        x : mygrad.Tensor
            Input data.

        Returns
        -------
        numpy.ndarray
            elementwise x/(1 + |x|).
        '''
        self.variables = (x,)
        return x.data / (1 + abs(x.data))

    def backward_var(self, grad, index, **kwargs):
        x = self.variables[index]
        x.backward(grad * x.data / ((1 + abs(x.data)) * (1 + abs(x.data))), **kwargs)

def softmax(x):
    ''' Returns the soft sign function x/(1 + |x|) elementwise along x

    Parameters
    ----------
    x : mygrad.Tensor
        Input data.

    Returns
    -------
    numpy.ndarray
        elementwise x/(1 + |x|).
    '''
    return Tensor._op(SoftSign, x)

