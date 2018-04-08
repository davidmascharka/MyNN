import numpy as np
from mygrad import Tensor
from mygrad.operations import Operation

class Hardtanh(Operation):
    ''' Returns the Hard(Saturated) Hyperbolic Tangent activation elementwise along x. The Hardtanh(x)
    is given by max(-1, min(1,x))
    '''
    def __call__(self, x):
        '''
        Parameters
        ----------
        x : mygrad.Tensor
            Input data.
        
        Returns
        -------
        numpy.ndarray
            elementwise max(min_val, min(max_val,x))
        '''    
        self.variables = (x,)
        return np.maximum(-1, np.minimum(1, x.data))

    def backward_var(self, grad, index, **kwargs):
        x = self.variables[index]
        x.backward(grad * np.piecewise(x.data, [x.data <= -1, x.data >= 1, x.data > -1 & x.data < 1], [0, 0, 1]), **kwargs)

def hardtanh(x):
    ''' Returns the Hard(Saturated) Hyperbolic Tangent activation elementwise along x. The Hardtanh(x)
    is given by max(-1, min(1,x))

    Parameters
    ----------
    x : mygrad.Tensor
        Input data.

    Returns
    -------
    numpy.ndarray
        elementwise max(-1, min(1,x))
    '''
    return Tensor._op(Hardtanh, x)