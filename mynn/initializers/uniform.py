import numpy as np
from mygrad import Tensor

def uniform(lower_bound=0, upper_bound=1, shape=None):
    ''' Initialize a :class:`mygrad.Tensor` by drawing from a uniform distribution.

    Parameters
    ----------
    lower_bound : Real, optional (default=0)
        Lower bound on the output interval, inclusive.

    upper_bound : Real, optional (default=1)
        Upper bound on the output interval, exclusive.

    shape : Tuple[int], optional (default=None)
        The output shape. If `None`, then the output is of shape ().

    Returns
    -------
    mygrad.Tensor, shape=`shape`
        A Tensor, with values drawn uniformly from [lower_bound, upper_bound).
    '''
    return Tensor(np.random.uniform(lower_bound, upper_bound, shape))
