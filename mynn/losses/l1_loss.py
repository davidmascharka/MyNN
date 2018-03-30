from mygrad.operations import Operation
from mygrad import Tensor
import numpy as np

class L1Loss(Operation):
    ''' Returns the L¹ loss Σ|xᵢ - yᵢ| averaged over the number of data points. '''
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
        The average L¹ loss.

        Extended Description
        --------------------
        The L1 loss is given by
        
        .. math::
            \frac{1}{N}\sum\limits_{1}^{N}|x_i - y_i|

        where :math:`N` is the number of elements in `x` and `y`.
        '''
        self.variables = (outputs,)
        outs = outputs.data
        loss = np.mean(np.abs(outs - targets))

        self.back = np.sign(outs - targets) / outs.shape[0]
        return loss

    def backward_var(self, grad, index, **kwargs):
        self.variables[index].backward(grad * self.back, **kwargs)

def l1_loss(x, y):
    ''' Returns the L¹ loss Σ|xᵢ - yᵢ| averaged over the number of data points. 

    Parameters
    ----------
    x : mygrad.Tensor, shape=(N,)
        The predictions for each of the N pieces of data.

    y : numpy.ndarray, shape=(N,)
        The correct value for each of the N pieces of data.

    Returns
    -------
    The average L¹ loss.

    Extended Description
    --------------------
    The L1 loss is given by

    .. math::
        \frac{1}{N}\sum\limits_{1}^{N}|x_i - y_i|

    where :math:`N` is the number of elements in `x` and `y`.
    '''
    return Tensor._op(L1Loss, x, op_args=(y,))
