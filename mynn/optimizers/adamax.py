import numpy as np

class AdaMax:
    ''' Performs the AdaMax variant of the Adam optimization procedure from Kingma and Ba.

    Parameters
    ----------
    params : Iterable
        The parameters over which to optimize.

    learning_rate : float, optional (default=0.002)
        Alpha in the original algorithm, this is the step size.

    beta1 : float ∈ [0, 1), optional (default=0.9)
        Decay rate for the first moment estimate (mean of the gradient).

    beta2 : float ∈ [0, 1), optional (default=0.999)
        Decay rate for the second moment estimate (uncentered variance of the gradient).

    eps : float, optional (default=1e-08)
        Epsilon value for numerical stability.
        

    Extended Description
    --------------------
    This optimizer implements the AdaMax variant of the Adam procedure described in the paper
      "Adam: A Method for Stochastic Optimization"
      Diederik P. Kingma and Jimmy Ba
    https://arxiv.org/abs/1412.6980
    '''
    def __init__(self, params, *, learning_rate=0.002, beta1=0.9, beta2=0.999, eps=1e-08):
        raise NotImplementedError
