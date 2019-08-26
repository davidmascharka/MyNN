from mygrad import mean, abs


def l1_loss(outputs, targets):
    """ Returns the L¹ loss Σ|xᵢ - yᵢ| averaged over the number of data points. 

    Parameters
    ----------
    outputs : mygrad.Tensor, shape=(N,)
        The predictions for each of the N pieces of data.

    targets : numpy.ndarray, shape=(N,)
        The correct value for each of the N pieces of data.

    Returns
    -------
    mygrad.Tensor, shape=()
        The average L¹ loss.

    Extended Description
    --------------------
    The L1 loss is given by
    
    .. math::
        \frac{1}{N}\sum\limits_{1}^{N}|x_i - y_i|

    where :math:`N` is the number of elements in `x` and `y`.
    """
    return mean(abs(outputs - targets))
