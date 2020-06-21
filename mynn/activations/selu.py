import numpy as np

from mygrad import Tensor
from mygrad.operation_base import Operation

__all__ = ["selu"]


class SELU(Operation):
    """ Returns the scaled exponential linear activation (SELU) elementwise along x. The SELU is
    given by  λɑ(exp(x) - 1) for x < 0 and λx for x ≥ 0.

    Notes
    -----
    The SELU activation was proposed in the paper
        Self-Normalizing Neural Networks
        Günter Klambauer, Thomas Unterthiner, Andreas Mayr, Sepp Hochreiter
    at https://arxiv.org/abs/1706.02515
    """

    def __call__(self, x):
        """
        Parameters
        ----------
        x : mygrad.Tensor
            Input data.

        Returns
        -------
        numpy.ndarray
            The SELU function applied to `x` elementwise.
        """
        self.variables = (x,)

        x = x.data
        self.alpha = 1.6732632423543772848170429916717
        self.scale = 1.0507009873554804934193349852946
        self.exp = self.alpha * (np.exp(x) - 1)
        return self.scale * np.where(x < 0, self.exp, x)

    def backward_var(self, grad, index, **kwargs):
        x = self.variables[index]
        x.backward(
            grad * self.scale * np.where(x.data < 0, self.exp + self.alpha, 1), **kwargs
        )


def selu(x):
    """ Returns the scaled exponential linear activation (SELU) elementwise along x. The SELU is
    given by  λɑ(exp(x) - 1) for x < 0 and λx for x ≥ 0.

    Parameters
    ----------
    x : mygrad.Tensor
        Input data.

    Returns
    -------
    mygrad.Tensor
        The SELU function applied to `x` elementwise.

    Notes
    -----
    The SELU activation was proposed in the paper
        Self-Normalizing Neural Networks
        Günter Klambauer, Thomas Unterthiner, Andreas Mayr, Sepp Hochreiter
    at https://arxiv.org/abs/1706.02515
    """
    return Tensor._op(SELU, x)
