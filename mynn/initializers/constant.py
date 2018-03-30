import numpy as np
from mygrad import Tensor

def constant(value=0, shape=None):
    ''' Initialize a :class:`mygrad.Tensor` of shape `shape` with a constant value.

    Parameters
    ----------
    value : Real, optional (default=0)
        The value with which to fill the Tensor.

    shape : Tuple[int], optional(default=None)
        The output shape. If `None`, then the output is of shape ().

    Returns
    -------
    mygrad.Tensor, shape=`shape`
        A Tensor, whose values are `value`.
    '''
    return Tensor(np.full(shape, value))
