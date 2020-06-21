from mygrad.nnet.activations import relu as mygrad_relu

__all__ = ["relu"]


def relu(x):
    """ Returns the rectified linear activation max(x, 0) elementwise along x.

    Parameters
    ----------
    x : mygrad.Tensor
        Input data.

    Returns
    -------
    mygrad.Tensor
        The rectified `x` (elementwise max(x, 0)).
    """
    return mygrad_relu(x, constant=False)
