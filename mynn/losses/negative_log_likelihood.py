from mygrad.operations import Operation
from mygrad import Tensor
import numpy as np

class NegativeLogLikelihood(Operation):
    ''' Returns the (weighted) negative log-likelihood loss between outputs and targets.
    
    Note that this does not compute a softmax, so you should input log-probabilities to this.
    See :class:`CrossEntropyLoss` if you need your loss to compute a softmax.
    '''
    def __call__(self, outputs, targets, weights=None):
        '''
        Parameters
        ----------
        outputs : mygrad.Tensor, shape=(N, C)
            The C log probabilities for each of the N pieces of data.
        
        targets : Sequence[int]
            The correct class indices, in [0, C), for each datum.

        weights : Sequence[Real], optional (default=None)
            The weighting factor to use on each class, or None.
        
        Returns
        -------
        The average (weighted) negative log-likelihood.
        '''
        if isinstance(targets, Tensor):
            targets = targets.data

        if isinstance(weights, Tensor):
            weights = weights.data
            
        self.variables = (outputs,)
        scores = outputs.data
        label_locs = (range(len(scores)), targets)
        factors = weights[targets] if weights is not None else np.ones_like(targets)
        total_weight = np.sum(factors)

        loss = -np.sum(scores[label_locs] * factors) / total_weight

        self.back = np.zeros_like(scores)
        self.back[label_locs] -= factors / total_weight
        return loss

    def backward_var(self, grad, index, **kwargs):
        self.variables[index].backward(grad * self.back, **kwargs)

def negative_log_likelihood(x, y, weights=None):
    '''
    Parameters
    ----------
    x : mygrad.Tensor, shape=(N, C)
        The C log probabilities for each of the N pieces of data.
    
    y : Sequence[int]
        The correct class indices, in [0, C), for each datum.

    weights : Sequence[Real], optional (default=None)
        The weighting factor to use on each class, or None.
    
    Returns
    -------
    The average (weighted) negative log-likelihood.
    '''
    return Tensor._op(NegativeLogLikelihood, x, op_args=(y, weights))
