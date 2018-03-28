from mygrad.operations import Operation
from mygrad import Tensor
import numpy as np

class MeanSquaredLoss(Operation):
    ''' Returns the mean squared error Σ(xᵢ - yᵢ)² over the data points '''
    def __call__(self, outputs, targets):
        '''
        Parameters
        ----------
        outputs : mygrad.Tensor, shape=(N, any)
            The model outputs for each of the N pieces of data.

        targets : numpy.ndarray, shape=(N, any)
            The correct value for each of the N pieces of data.

        Returns
        -------
        The mean squared error.
        '''
        raise NotImplementedError


    def backward_a(self, grad):
        raise NotImplementedError
