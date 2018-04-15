from mygrad import abs

def soft_sign(x):
    ''' Returns the soft sign function x/(1 + |x|).

    Parameters
    ----------
    x : mygrad.Tensor
        Input data.

    Returns
    -------
    numpy.ndarray
        The soft sign function applied to `x` elementwise.
    '''
    return x / (1 + abs(x))
