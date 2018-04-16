from mygrad import maximum

__all__ = ['relu']

def relu(x):
    ''' Returns the rectified linear activation max(x, 0) elementwise along x.

    Parameters
    ----------
    x : mygrad.Tensor
        Input data.

    Returns
    -------
    numpy.ndarray
        The rectified `x` (elementwise max(x, 0)).
    '''
    return maximum(x, 0)
