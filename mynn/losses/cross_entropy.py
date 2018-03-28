from mygrad.operations import Operation
from mygrad import Tensor
import numpy as np

class CrossEntropyLoss(Operation):
    ''' Returns the cross-entropy loss between outputs and targets. '''
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
        The average cross-entropy loss.
        '''
        raise NotImplementedError

    def backward_a(self, grad):
        raise NotImplementedError
