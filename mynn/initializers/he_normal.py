import numpy as np
from mygrad import Tensor


def he_normal(*shape, gain=1):
    """ Initialize a :class:`mygrad.Tensor` according to the normal initialization procedure
    described by He et al.

    Parameters
    ----------
    shape : Sequence[int]
        The shape of the output Tensor. Note that `shape` must be at least two-dimensional.

    gain : Real, optional (default=1)
        The gain (scaling factor) to apply.

    Returns
    -------
    mygrad.Tensor, shape=`shape`
        A Tensor, with values initialized according to the He normal initialization.

    Extended Description
    --------------------
    He, Zhang, Ren, and Sun put forward this initialization in the paper
        "Delving Deep into Rectifiers: Surpassing Human-Level Performance
        on ImageNet Classification"
    https://arxiv.org/abs/1502.01852

    A Tensor :math:`W` initialized in this way should be drawn from a distribution about

    .. math::
        \mathcal{N}(0, \sqrt{\frac{2}{(1+a^2)n_l}})

    where :math:`a` is the slope of the rectifier following this layer, which is incorporated
    using the `gain` variable above.
    """
    if len(shape) < 2:
        raise ValueError("He Normal initialization requires at least two dimensions")

    tensor = np.empty(shape)
    std = gain / np.sqrt(shape[1] * tensor[0, 0].size)
    return Tensor(np.random.normal(0, std, shape))
