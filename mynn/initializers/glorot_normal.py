import numpy as np
from mygrad import Tensor


def glorot_normal(*shape, gain=1):
    """ Initialize a :class:`mygrad.Tensor` according to the normal initialization procedure
    described by Glorot and Bengio.

    Parameters
    ----------
    shape : Sequence[int]
        The shape of the output Tensor. Note that `shape` must be at least two-dimensional.

    gain : Real, optional (default=1)
        The gain (scaling factor) to apply.

    Returns
    -------
    mygrad.Tensor, shape=`shape`
        A Tensor, with values initialized according to the glorot normal initialization.

    Extended Description
    --------------------
    Glorot and Bengio put forward this initialization in the paper
        "Understanding the Difficulty of Training Deep Feedforward Neural Networks"
    http://proceedings.mlr.press/v9/glorot10a/glorot10a.pdf

    A Tensor :math:`W` initialized in this way should be drawn from a distribution about

    .. math::
        \mathcal{N}(0, \frac{\sqrt{2}}{\sqrt{n_j+n_{j+1}}})
    """
    if len(shape) < 2:
        raise ValueError("Glorot Normal initialization requires at least two dimensions")

    tensor = np.empty(shape)
    std = gain * np.sqrt(2 / ((shape[0] + shape[1]) * tensor[0, 0].size))
    return Tensor(np.random.normal(0, std, shape))
