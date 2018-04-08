import numpy as np
from mygrad import Tensor
from mygrad.operations import Operation
from numpy import ones,vstack
from numpy.linalg import lstsq

class Hardtanh(Operation):
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
        self.min_val = min_val
        self.max_val = max_val
        points = [(max_val, 1),(min_val,-1)]
        x_coords, y_coords = zip(*points)
        A = vstack([x_coords,ones(len(x_coords))]).T
        m, b = lstsq(A, y_coords)[0]
        self.slope = b
        return np.maximum(-1, np.minimum(1, (x.data * m) + self.slope))

    def backward_var(self, grad, index, **kwargs):
        x = self.variables[index]
        x.backward(grad * np.piecewise(x.data, [x.data <= min_val, x.data >= max_val, x.data > min_val & x.data < max_val], [0, 0, self.slope]), **kwargs)

def hardtanh(x, min_val = -1, max_val = 1):
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
    return Tensor._op(Hardtanh, x, op_args=(min_val,max_val))
