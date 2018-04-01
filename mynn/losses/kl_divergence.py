from mygrad.operations import Operation
from mygrad import Tensor
import numpy as np

class KLDivergenceLoss(Operation):
    ''' Returns the Kullback-Leibler divergence loss from the outputs to the targets.
    
    The KL-Divergence loss for a single sample is given by yᵢ⊙(log(yᵢ) - xᵢ)
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
        if isinstance(targets, Tensor):
            targets = targets.data

        self.variables = (outputs,)
        loss = np.mean(targets * (np.log(targets) - outputs.data))

        self.back = -np.maximum(targets, 0) / outputs.size
        return loss

    def backward_var(self, grad, index, **kwargs):
        self.variables[index].backward(grad * self.back, **kwargs)

def kl_divergence(x, y):
    '''
    Parameters
    ----------
    outputs : mygrad.Tensor, shape=(N, any)
        The model outputs for each of the N pieces of data.

    targets : numpy.ndarray, shape=(N, any)
        The correct vaue for each datum.
    '''
    return Tensor._op(KLDivergenceLoss, x, op_args=(y,))
