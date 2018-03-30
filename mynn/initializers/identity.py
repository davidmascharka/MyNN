import numpy as np
from mygrad import Tensor

def identity(size=3):
    ''' Initialize the identity matrix of dimension `size`x`size`.

    Parameters
    ----------
    size : int, optional (default=3)
        The size of the identity matrix.

    Returns
    mygrad.Tensor, shape=(`size`, `size`)
        A Tensor, with ones on the main diagonal and zeros elsewhere.
    '''
    return Tensor(np.identity(size))
