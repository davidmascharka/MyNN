import numpy as np
from mygrad import Tensor


def constant(*shape, value=0):
    """ Initialize a :class:`mygrad.Tensor` of shape `shape` with a constant value.

    Parameters
    ----------
    shape : Sequence[int]
        The output shape.

    value : Real, optional (default=0)
        The value with which to fill the Tensor.

    Returns
    -------
    mygrad.Tensor, shape=`shape`
        A Tensor, whose values are `value`.
    """
    return Tensor(np.full(shape, value))
