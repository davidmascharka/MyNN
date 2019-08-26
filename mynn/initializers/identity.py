import numpy as np
from mygrad import Tensor


def identity(size):
    """ Initialize the identity matrix of dimension `size`x`size`.

    Parameters
    ----------
    size : int
        The size of the identity matrix.

    Returns
    mygrad.Tensor, shape=(`size`, `size`)
        A Tensor, with ones on the main diagonal and zeros elsewhere.
    """
    return Tensor(np.identity(size))
