from mygrad.operations import Operation
from mygrad import Tensor
import numpy as np

class FocalLoss(Operation):
    ''' Returns the focal loss as described in https://arxiv.org/abs/1708.02002 

    Parameters
    ----------
    gamma : Real
        The focal factor.
    '''
    def __init__(self, gamma):
        self.gamma = gamma

    def __call__(self, outputs, targets):
        '''
        Parameters
        ----------
        outputs : mygrad.Tensor, shape=(N, C)
            The C class scores for each of the N pieces of data.

        targets : Sequence[int]
            The correct class indices, in [0, C), for each datum.

        Returns
        -------
        The average focal loss.
        '''
        raise NotImplementedError

    def backward_a(self, grad):
        raise NotImplementedError
