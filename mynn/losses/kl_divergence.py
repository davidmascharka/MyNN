from mygrad.operations import Operation
from mygrad import Tensor
import numpy as np

class KLDivergenceLoss(Operation):
    ''' Returns the Kullback-Leibler divergence loss from the outputs to the targets.
    
    The KL-Divergence loss for a single sample is given by yᵢ*(log(yᵢ) - xᵢ)
    '''
    def __call__(self, outputs, targets):
        '''
        Parameters
        ----------
        outputs : mygrad.Tensor, shape=(N, any)
            The model outputs for each of the N pieces of data.

        targets : numpy.ndarray, shape=(N, any)
            The correct vaue for each datum.
        '''
        raise NotImplementedError

    def backward_a(self, grad):
        raise NotImplementedError
