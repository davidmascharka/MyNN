import numpy as np


class RMSprop:
    """ Performs the rmsprop optimization procedure from Hinton.

    Parameters
    ----------
    params : Iterable
        The parameters over which to optimize.

    learning_rate : float, optional (default=0.01)
        The step size.

    alpha : float ∈ [0, 1), optional (default=0.9)
        A smoothing factor.

    eps : float, optional (default=1e-08)
        Epsilon term to improve numerical stability.

    Extended Description
    --------------------
    This optimizer implements the rmsprop procedure described in the lecture notes
      "Lecture 6e rmsprop: Divide the gradient by a running average of its recent magnitude
      Geoff Hinton with Nitish Srivastava and Kevin Swersky
    http://www.cs.toronto.edu/~tijmen/csc321/slides/lecture_slides_lec6.pdf
    """

    def __init__(self, params, *, learning_rate=0.01, alpha=0.9, eps=1e-08):
        raise NotImplementedError
