import numpy as np

class Adadelta:
    ''' Performs the Adadelta optimization procedure from Zeiler.

    Parameters
    ----------
    params : Iterable
        The parameters over which to optimize.

    rho : float ∈ [0, 1), optional (default=0.95)
        The decay rate in the running average.

    eps : float, optional (default=1e-06)
        Epsilon value to better condition the denominator.


    Extended Description
    --------------------
    This optimizer implements the Adadelta optimization procedure described in the paper
      "ADADELTA: An Adaptive Learning Rate Method"
      Matthew Zeiler
    https://arxiv.org/abs/1212.5701
    '''
    def __init__(self, params, *, rho=0.95, eps=1e-06):
        raise NotImplementedError
