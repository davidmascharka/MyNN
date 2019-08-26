import numpy as np


class dropout:
    """ A dropout layer

    This layer will randomly set elements of a tensor to 0, with some specified probability p.

    Those elements that pass through unmasked are scaled by a factor of  1 / (1 - p), such that
    the magnitude of the tensor, in expectation, is unaffected by the dropout layer.

    Examples
    --------
    >>> import mygrad as mg
    >>> x = mg.Tensor([1, 2, 3])

    "Dropout" each element of `x` with probability p=0.5. The other elements
    are scaled by 1 / (1-p) so that the magnitude of `x`, on average, remains
    unchanged.

    >>> dropout(x, prob_dropout=0.5)
    Tensor([2., 0., 6.])

    >>> dropout(x, prob_dropout=0.5)
    Tensor([0., 0., 6.])

    >>> dropout(x, prob_dropout=0.5)
    Tensor([2., 4., 0.])
    """

    def __init__(self, prob_dropout):
        """ Parameters
            ----------
            prob_dropout : float
                The probability, specified on [0, 1), that any given element of the input
                will be masked (i.e. set to 0).

                If `prob_dropout=0` the tensor is passed through unchanged."""
        assert 0 <= prob_dropout < 1
        self.p_dropout = prob_dropout

    def __call__(self, x):
        """ Perform the dropout on `x`

        Parameters
        ----------
        x : Union[numpy.ndarray, mygrad.Tensor]
            The data whose elements will each be dropped with probability p.

        Returns
        -------
        Union[numpy.ndarray, mygrad.Tensor]
            `x` with dropout applied.
        """
        if not self.p_dropout:
            return x
        drop_mask = np.random.binomial(1, (1 - self.p_dropout), x.shape)
        return x * drop_mask / (1 - self.p_dropout)

    @property
    def parameters(self):
        """ Access the parameters of the layer.

        Returns
        -------
        Tuple[]
            dropout is a parameterless layer
        """
        return tuple()
