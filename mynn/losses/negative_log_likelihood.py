from mygrad.operations import Operation
from mygrad import Tensor
import numpy as np

class NegativeLogLikelihood(Operation):
    ''' Returns the negative log-likelihood. '''
    def __call__(self, outputs, targets):
        '''
        Parameters
        ----------
        outputs : mygrad.Tensor, shape=(N, C)
            The C log probabilities for each of the N pieces of data.
        
        targets : Sequence[int]
            The correct class indices, in [0, C), for each datum.
        
        Returns
        -------
        The average negative log-likelihood.
        '''
        raise NotImplementedError

    def backward_a(self, grad):
        raise NotImplementedError
