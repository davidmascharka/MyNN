from mygrad import Tensor, sum
import numpy as np


def negative_log_likelihood(outputs, targets, *, weights=None):
    """ Returns the (weighted) negative log-likelihood loss between outputs and targets.
    
    Note that this does not compute a softmax, so you should input log-probabilities to this.
    See ``softmax_cross_entropy`` if you need your loss to compute a softmax.

    Parameters
    ----------
    outputs : mygrad.Tensor, shape=(N, C)
        The C log probabilities for each of the N pieces of data.
    
    targets : Union[mygrad.Tensor, Sequence[int]], shape=(N,)
        The correct class indices, in [0, C), for each datum.

    weights : Union[mygrad.Tensor, Sequence[Real]], optional (default=None)
        The weighting factor to use on each class, or None.
    
    Returns
    -------
    mygrad.Tensor, shape=()
        The average (weighted) negative log-likelihood.
    """
    if isinstance(targets, Tensor):
        targets = targets.data

    label_locs = (range(len(targets)), targets)
    factors = weights[targets] if weights is not None else np.ones_like(targets)
    return -sum(outputs[label_locs] * factors) / sum(factors)
