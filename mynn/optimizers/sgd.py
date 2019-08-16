import numpy as np


class SGD:
    """ Performs (batched) stochastic gradient descent.

    Parameters
    ----------
    params : Iterable
        The parameters over which to optimize.

    learning_rate : Real, optional (default=0.1)
        The step size.

    momentum : Real ∈ [0, 1), optional (default=0)
        The momentum term.

    weight_decay : Real, optional (default=0)
        The weight decay term.
    """

    def __init__(self, params, *, learning_rate=0.1, momentum=0, weight_decay=0):
        if momentum < 0 or momentum > 1:
            raise ValueError("Momentum must lie within [0, 1)")

        self.params = params
        self.learning_rate = learning_rate
        self.momentum = momentum
        self.weight_decay = weight_decay

        self.param_moments = []
        if momentum != 0:
            for param in params:
                self.param_moments.append(np.zeros_like(param.data))

    def step(self):
        """ Perform one optimization step.

        This function should be called after accumulating gradients into the parameters of the model
        you wish to optimize via `backward()`. This will perform one step of (stochatic) gradient
        descent, optionally with momentum. The update here is given by

          ν = μν - η(dθ)
          θ = θ + ν

        where θ is the parameter list of the model, η is the learning rate, μ is the weighting on
        the momentum term, and ν is the moment of each parameter. Note that if the momentum term is
        0, this simplifies to θ = θ - η(dθ)
        """
        for idx, param in enumerate(self.params):
            if param.grad is None:
                continue

            # no decay on bias
            update = -self.weight_decay * param.data if param.ndim > 1 else 0

            # perform the momentum update
            if self.momentum != 0:
                moment = self.param_moments[idx]
                moment *= self.momentum
                moment -= self.learning_rate * param.grad
                update += moment
            else:
                update += -self.learning_rate * param.grad

            # update parameters
            param.data += update
