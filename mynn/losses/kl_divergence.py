from mygrad import mean, log


def kl_divergence(outputs, targets):
    """ Returns the Kullback-Leibler divergence loss from the outputs to the targets.
    
    The KL-Divergence loss for a single sample is given by yᵢ⊙(log(yᵢ) - xᵢ)

    Parameters
    ----------
    outputs : mygrad.Tensor, shape=(N, any)
        The model outputs for each of the N pieces of data.

    targets : numpy.ndarray, shape=(N, any)
        The correct vaue for each datum.

    Returns
    -------
    mygrad.Tensor, shape=()
        The mean Kullback-Leibler divergence.
    """
    return mean(targets * (log(targets) - outputs))
