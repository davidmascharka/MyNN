from mygrad.nnet.activations import softmax as mygrad_softmax

__all__ = ["softmax"]


def softmax(x):
    """ Returns the softmax exp(x) / Σexp(x).

    Parameters
    ----------
    x : mygrad.Tensor
        Input data.

    Returns
    -------
    mygrad.Tensor
        The softmax of `x`.
    """
    return mygrad_softmax(x, constant=False)
