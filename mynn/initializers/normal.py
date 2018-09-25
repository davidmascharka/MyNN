import numpy as np
from mygrad import Tensor

def normal(*shape, mean=0, std=1):
    """ Initialize a :class:`mygrad.Tensor` by drawing from a normal (Gaussian) distribution.

    Parameters
    ----------
    shape : Sequence[int]
        The output shape.

    mean : Real, optional (default=0)
        The mean of the distribution from which to draw.

    std : Real, optional (default=1)
        The standard deviation of the distribution from which to draw.

    Returns
    -------
    mygrad.Tensor, shape=`shape`
        A Tensor, with values drawn from Ɲ(μ, σ²), where μ=`mean` and σ=`std`.
    """
    return Tensor(np.random.normal(mean, std, shape))
