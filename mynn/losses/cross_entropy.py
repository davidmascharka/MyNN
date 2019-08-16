from mynn.activations.log_softmax import log_softmax
from mynn.losses.negative_log_likelihood import negative_log_likelihood
from mygrad.nnet.losses import softmax_crossentropy as mygrad_softmax_cross_entropy


def softmax_cross_entropy(x, y, *, weights=None):
    """
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
    mygrad.Tensor, shape=()
        The average cross-entropy loss.
    """
    if weights is not None:
        return negative_log_likelihood(log_softmax(x), y, weights=weights)
    return mygrad_softmax_cross_entropy(x, y)
