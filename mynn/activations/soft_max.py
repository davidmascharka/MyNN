import numpy as np
from mygrad import Tensor
from mygrad.operation_base import Operation

class Softmax(Operation):
    ''' Returns the SoftMax function exp(xᵢ) / ∑exp(xⱼ)'''
    def __call__(self, x):
        '''
        Parameters
        ----------
        x : mygrad.Tensor
            Input data.

        Returns
        -------
        numpy.ndarray
            elementwise exp(x) / ∑j exp(xj).
        '''
        self.variables = (x,)
        assert 0 < x.ndim < 3
        self.__kw = dict(axis=1, keepdims=True) if x.ndim == 2 else dict(axis=None, keepdims=False)
        x.data = x.data - x.data.max(**self.__kw)
        np.exp(x.data, out=x.data)
        x.data /= x.data.sum(**self.__kw)
        return x.data

    def backward_var(self, grad, index, **kwargs):
        soft = self(x.data)
        sg = soft * grad
        x.backward(sg - soft * np.sum(sg, **self.__kw), **kwargs)

def softmax(x):
    ''' Returns the SoftMax function elementwise along x given by exp(x) / ∑j exp(xj)

    Parameters
    ----------
    x : mygrad.Tensor
        Input data.

    Returns
    -------
    numpy.ndarray
            The softmax function is given by

            .. math::
                softmax(x_i) = \frac{e^{x_i}}{\sum\limits_1^N e^{x_j}}
    '''
    return Tensor._op(Softmax, x)
