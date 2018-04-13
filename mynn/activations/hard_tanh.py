import numpy as np
from mygrad import Tensor
from mygrad.operation_base import Operation
from mygrad import maximum, minimum

class HardTanh(Operation):
    ''' Returns the Hard(Saturated) Hyperbolic Tangent activation elementwise along x. The Hardtanh(x)
    is given by max(-1, min(1,x))
    '''
    def __call__(self, x, min_val, max_val):
        '''
        Parameters
        ----------
        x : mygrad.Tensor
            Input data.
    
        min_val : Real
            Minimum value of the linear region range. Default: -1
        
        max_val : Real
            Maximum value of the linear region range. Default: 1
        
        Returns
        -------
        numpy.ndarray
            elementwise max(min_val, min(max_val,x))
        '''
        assert max_val > min_val
        self.variables = (x,)
        self.back = np.logical_and(min_val < x, x < max_val)
        return maximum(min_val, minimum(x, max_val))

    def backward_var(self, grad, index, **kwargs):
        x = self.variables[index]
        x.backward(grad * self.back)

def hard_tanh(x, min_val = -1, max_val = 1):
    ''' Returns the Hard(Saturated) Hyperbolic Tangent activation elementwise along x. The Hardtanh(x)
    is given by max(-1, min(1,x))

    Parameters
    ----------
    x : mygrad.Tensor
        Input data.

    min_val : Real
        Minimum value of the linear region range. Default: -1
        
    max_val : Real
        Maximum value of the linear region range. Default: 1

    Returns
    -------
    numpy.ndarray
        elementwise max(min_val, min(max_val,x))
    '''
    assert max_val > min_val
    return Tensor._op(HardTanh, x, op_args=(min_val,max_val))
