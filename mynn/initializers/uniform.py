import numpy as np
from mygrad import Tensor


def uniform(*shape, lower_bound=0, upper_bound=1):
    """ Initialize a :class:`mygrad.Tensor` by drawing from a uniform distribution.

    Parameters
    ----------
    shape : Sequence[int]
        The output shape.

    lower_bound : Real, optional (default=0)
        Lower bound on the output interval, inclusive.

    upper_bound : Real, optional (default=1)
        Upper bound on the output interval, exclusive.

    Returns
    -------
    mygrad.Tensor, shape=`shape`
        A Tensor, with values drawn uniformly from [lower_bound, upper_bound).
    """
    return Tensor(np.random.uniform(lower_bound, upper_bound, shape))
