from mygrad import maximum, minimum

__all__ = ['leaky_relu']

def leaky_relu(x, slope):
    ''' Returns the leaky rectified linear activation elementwise along x. The leaky ReLU is given
    by max(x, 0) + slope*min(x, 0).

    Parameters
    ----------
    x : mygrad.Tensor
        Input data.

    slope : Real
        The slop of the negative activation.

    Returns
    -------
    numpy.ndarray
        The rectified `x` (elementwise max(x, 0)).
    '''
    return maximum(x, 0) + slope * minimum(x, 0)
