from mygrad import exp

__all__ = ['softmax']

def softmax(x):
    ''' Returns the softmax exp(x) / Σexp(x).

    Parameters
    ----------
    x : mygrad.Tensor
        Input data.

    Returns
    -------
    mygrad.Tensor
        The softmax of `x`.
    '''
    assert 0 < x.ndim < 3, 'Input must be 1- or 2-dimensional.'

    kw = dict(axis=1, keepdims=True) if x.ndim == 2 else dict(axis=None, keepdims=False)
    
    x = exp(x - x.max(**kw))
    return x / x.sum(**kw)
