from mygrad.nnet.activations import sigmoid as mygrad_sigmoid

__all__ = ["sigmoid"]


def sigmoid(x):
    """ Returns the sigmoid function of the input elementwise.

    Parameters
    ----------
    x : mygrad.Tensor
        The input.

    Returns
    -------
    mygrad.Tensor
        The input with the sigmoid function applied elementwise.

    Extended Description
    --------------------
    The sigmoid function is given by

    .. math::
        \sigma(x) = \frac{e^x}{e^x + 1}
    """
    return mygrad_sigmoid(x, constant=False)
