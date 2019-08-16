import numpy as np

from mygrad.operation_base import Operation
from mygrad import Tensor

__all__ = ["huber_loss"]


class HuberLoss(Operation):
    """ Returns the Huber loss (smooth L1).

    Extended Description
    --------------------
    The Huber loss is given by

    .. math::
        L_\delta(x, y) = \frac{1}{N}\sum\limits_1^N \bigl\{ \begin{array}{l l} 
            \frac{(x_i - y_i)^2}{2} & |x_i - y_i| \leq \delta\\
            \delta|x_i - y_i| - \frac{\delta}{2} & |x_i - y_i| > \delta\end{array}
    """

    scalar_only = True

    def __call__(self, outputs, targets, delta=1):
        """
        Parameters
        ----------
        outputs : mygrad.Tensor, shape=(N, any)
            The output for each of the N pieces of data.

        targets : Union[mygrad.Tensor, numpy.ndarray], shape=(N, any)
            The target for each datum.

        delta : Real > 0, optional (default=1)
            The value under which to use a squared error.

        Returns
        -------
        numpy.ndarray
            The average Huber loss.
        """
        if delta <= 0:
            raise ValueError("Delta must be > 0")

        if isinstance(targets, Tensor):
            targets = targets.data

        self.variables = (outputs,)
        outs = outputs.data

        diff = outs - targets
        sign = np.sign(diff)
        np.abs(diff, out=diff)

        loss = diff.copy()
        np.multiply(loss, diff / 2, where=(diff < 1), out=loss)
        loss[diff >= 1] -= 0.5 * delta
        loss[diff >= 1] *= delta

        self.back = np.where(diff < 1, outs - targets, delta * sign) / outs.size

        return np.mean(loss)

    def backward_var(self, grad, index, **kwargs):
        return grad * self.back


def huber_loss(x, y, *, delta=1):
    """ Returns the Huber loss (smooth L1).

    Parameters
    ----------
    outputs : mygrad.Tensor, shape=(N, any)
        The output for each of the N pieces of data.

    targets : Union[mygrad.Tensor, numpy.ndarray], shape=(N, any)
        The target for each datum.

    delta : Real > 0, optional (default=1)
        The value under which to use a squared error.

    Returns
    -------
    mygrad.Tensor, shape=()
        The average Huber loss.

    Extended Description
    --------------------
    The Huber loss is given by

    .. math::
        L_\delta(x, y) = \frac{1}{N}\sum\limits_1^N \bigl\{ \begin{array}{l l} 
            \frac{(x_i - y_i)^2}{2} & |x_i - y_i| \leq \delta\\
            \delta|x_i - y_i| - \frac{\delta}{2} & |x_i - y_i| > \delta\end{array}

    """
    return Tensor._op(HuberLoss, x, op_args=(y, delta))
