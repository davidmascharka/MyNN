from mygrad.operations import Operation
from mygrad import Tensor
import numpy as np

class MeanSquaredLoss(Operation):
    ''' Returns the mean squared error Σ(xᵢ - yᵢ)² over the data points. '''
    def __call__(self, outputs, targets):
        '''
        Parameters
        ----------
        outputs : mygrad.Tensor, shape=(N,)
            The predictions for each of the N pieces of data.

        targets : numpy.ndarray, shape=(N,)
            The correct value for each of the N pieces of data.

        Returns
        -------
        The mean squared error.

        Extended Description
        --------------------
        The mean squared error is given by

        .. math::
            \frac{1}{N}\sum\limits_{1}^{N}(x_i - y_i)^2

        where :math:`N` is the number of elements in `x` and `y`.
        '''
        if isinstance(targets, Tensor):
            targets = targets.data
            
        self.variables = (outputs,)
        outs = outputs.data

        tmp = outs - targets
        self.back = 2 * tmp / outs.shape[0]
        loss = np.sum(tmp ** 2)
        return loss


    def backward_var(self, grad, index, **kwargs):
        self.variables[index].backward(grad * self.back, **kwargs)

def mean_squared_loss(x, y):
    ''' Returns the mean squared error Σ(xᵢ - yᵢ)² over the data points.

    Parameters
    ----------
    outputs : mygrad.Tensor, shape=(N,)
        The predictions for each of the N pieces of data.

    targets : numpy.ndarray, shape=(N,)
        The correct value for each of the N pieces of data.

    Returns
    -------
    The mean squared error.

    Extended Description
    --------------------
    The mean squared error is given by

    .. math::
        \frac{1}{N}\sum\limits_{1}^{N}(x_i - y_i)^2

    where :math:`N` is the number of elements in `x` and `y`.
    '''
    return Tensor._op(MeanSquaredLoss, x, op_args=(y,))
