from mygrad import abs

__all__ = ["soft_sign"]


def soft_sign(x):
    """ Returns the soft sign function x/(1 + |x|).

    Parameters
    ----------
    x : mygrad.Tensor
        Input data.

    Returns
    -------
    mygrad.Tensor
        The soft sign function applied to `x` elementwise.
    """
    return x / (1 + abs(x))
