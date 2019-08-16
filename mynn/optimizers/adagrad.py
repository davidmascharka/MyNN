import numpy as np


class AdaGrad:
    """ Performs the AdaGrad optimization procedure from Duchi, Hazan, and Singer.

    Parameters
    ----------
    params : Iterable
        The parameters over which to optimize.

    learning_rate : float, optional (default=0.01)
        The step size; eta in the paper.

    Extended Description
    --------------------
    This optimizer implements the AdaGrad procedure described in the paper
      "Adaptive Subgradient Methods for Online Learning and Stochastic Optimization"
      John Duchi, Elad Hazan, and Yoram Singer
    http://jmlr.org/papers/volume12/duchi11a/duchi11a.pdf
    """

    def __init__(self, params, *, learning_rate=0.01):
        raise NotImplementedError
