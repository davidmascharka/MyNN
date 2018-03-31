from mygrad.operations import Operation
from mygrad import Tensor
import numpy as np

class CrossEntropyLoss(Operation):
    ''' Returns the (weighted) cross-entropy loss between outputs and targets.

    Note that this computes the softmax, and so should not take log-probabilities. If you
    have log-probs, you should instead use :class:`NegativeLogLikelihood`.
    '''
    def __call__(self, outputs, targets, weights=None):
        '''
        Parameters
        ----------
        outputs : mygrad.Tensor, shape=(N, C)
            The C class scores for each of the N pieces of data.

        targets : Sequence[int]
            The correct class indices, in [0, C), for each datum.

        weights : Sequence[Real], optional (default=None)
            The weighting factor to use on each class, or None.
       
        Returns
        -------
        The average (weighted) cross-entropy loss.
        '''
        self.variables = (outputs,)
        scores = np.copy(outputs.data)
        max_scores = np.max(scores, axis=1, keepdims=True)
        np.exp(scores - max_scores, out=scores)
        scores /= np.sum(scores, axis=1, keepdims=True)
        label_locs = (range(len(scores)), targets)
        factors = weights[targets] if weights is not None else np.ones_like(targets)
        total_weight = np.sum(factors)
        
        loss = -np.sum(np.log(scores[label_locs]) * factors) / total_weight

        self.back = scores
        self.back[label_locs] -= 1
        self.back *= factors.reshape(-1, 1) / total_weight
        return loss

    def backward_var(self, grad, index, **kwargs):
        self.variables[index].backward(grad * self.back, **kwargs)

def cross_entropy(x, y, weights=None):
    '''
    Parameters
    ----------
    x : mygrad.Tensor, shape=(N, C)
        The C class scores for each of the N pieces of data.

    y : Sequence[int]
        The correct class indices, in [0, C), for each datum.

    weights : Sequence[Real], optional (default=None)
        The weighting factor to use on each class, or None.
   
    Returns
    -------
    The average cross-entropy loss.
    '''
    return Tensor._op(CrossEntropyLoss, x, op_args=(y, weights))
