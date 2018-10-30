from mygrad import maximum, minimum

__all__ = ['hard_tanh']

def hard_tanh(x, *, lower_bound=-1, upper_bound=1):
    """ Returns the hard hyperbolic tangent function.

    The hard_tanh function is `lower_bound` where `x` <= `lower_bound`, `upper_bound` where 
    `x` >= `upper_bound`, and `x` where `lower_bound` < `x` < `upper_bound`.

    Parameters
    ----------
    x : Union[numpy.ndarray, mygrad.Tensor]
        The input, to which to apply the hard tanh function.

    lower_bound : Real, optional (default=-1)
        The lower bound on the hard tanh.

    upper_bound : Real, optional (default=1)
        The upper bound on the hard tanh.

    Returns
    -------
    mygrad.Tensor
        The result of applpying the hard tanh function elementwise to `x`.
    """
    return maximum(lower_bound, minimum(x, upper_bound))
