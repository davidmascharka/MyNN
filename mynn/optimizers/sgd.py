import numpy as np

class SGD:
    ''' Performs (batched) stochastic gradient descent.

    Parameters
    ----------
    params : Iterable
        The parameters over which to optimize.

    learning_rate : float, optional (default=0.1)
        The step size.

    momentum : float ∈ [0, 1), optional (default=0.0)
        The momentum term.
    '''
    def __init__(self, params, *, learning_rate=0.1, momentum=0.0):
        raise NotImplementedError
