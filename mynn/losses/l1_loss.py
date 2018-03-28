from mygrad.operations import Operation
from mygrad import Tensor
import numpy as np

class L1Loss(Operation):
    ''' Returns the L¹ loss Σ|xᵢ - yᵢ| averaged over the number of data points '''
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
        The average L¹ loss.
        '''
        raise NotImplementedError


    def backward_a(self, grad):
        raise NotImplementedError
