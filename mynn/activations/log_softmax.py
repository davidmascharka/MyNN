from mygrad.nnet.activations import logsoftmax as mygrad_log_softmax

__all__ = ['log_softmax']

def log_softmax(x):
    ''' Returns the log softmax:
          log(exp(x) / Σexp(x)).

    However, this implementation is faster and more numerically-stable
    than performing the softmax followed by the log.

    Parameters
    ----------
    x : mygrad.Tensor
        Input data.

    Returns
    -------
    mygrad.Tensor
        The log-softmax of `x`.
    '''
    return mygrad_log_softmax(x)
