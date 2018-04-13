from mygrad import Tensor, log, exp, sum

def log_softmax(x):
    ''' Returns the log softmax log(exp(x) / Σexp(x)). However, this implementation is faster and
    more numerically-stable than performing the softmax followed by the log.

    Parameters
    ----------
    x : mygrad.Tensor
        Input data.

    Returns
    -------
    numpy.ndarray
        The log-softmax of `x`.
    '''
    assert 0 < x.ndim < 3, 'Input must be 1- or 2-dimensional.'

    kw = dict(axis=1, keepdims=True) if x.ndim == 2 else dict(axis=None, keepdims=False)
    
    x = x - x.max(**kw)
    return x - log(exp(x).sum(**kw))
