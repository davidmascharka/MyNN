import numpy as np
from mygrad import Tensor
from mygrad.operations import Operation

class SoftMax(Operation):
    ''' Returns the SoftMax function elementwise along x given by exp(x) / ∑j exp(xj)'''
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
        shiftx = x.data - np.max(x.data)
        exps = np.exp(shiftx)
        self.value = exps / np.sum(exps)
        SM = self.value.reshape((-1,1))
        self.jac = np.diag(self.value) - np.dot(SM, SM.T)
        return self.value

    def backward_var(self, grad, index, **kwargs):
        x = self.variables[index]
        x.backward(grad * self.jac[index], **kwargs)

def softmax(x):
    ''' Returns the SoftMax function elementwise along x given by exp(x) / ∑j exp(xj)

    Parameters
    ----------
    x : mygrad.Tensor
        Input data.

    Returns
    -------
    numpy.ndarray
            elementwise exp(x) / ∑j exp(xj).
    '''
    return Tensor._op(SoftMax, x)
