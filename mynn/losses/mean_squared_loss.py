from mygrad import mean


def mean_squared_loss(outputs, targets):
    """ Returns the mean squared error Σ(xᵢ - yᵢ)² over the data points. 

    Parameters
    ----------
    outputs : mygrad.Tensor, shape=(N, any)
        The model outputs, where `N` is the number of items.

    targets : mygrad.Tensor, shape=(N, any)
        The target values, where `N` is the number of items.

    Returns
    -------
    mygrad.Tensor, shape=()
        The mean squared error.

    Extended Description
    --------------------
    The mean squared error is given by

    .. math::
        \frac{1}{N}\sum\limits_{1}^{N}(x_i - y_i)^2

    where :math:`N` is the number of elements in `x` and `y`.
    """
    return mean((outputs - targets) ** 2)
