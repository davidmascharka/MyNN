import numpy as np


class AdaMax:
    """ Performs the AdaMax variant of the Adam optimization procedure from Kingma and Ba.

    Parameters
    ----------
    params : Iterable
        The parameters over which to optimize.

    learning_rate : float, optional (default=0.002)
        Alpha in the original algorithm, this is the step size.

    beta1 : float ∈ [0, 1), optional (default=0.9)
        Decay rate for the first moment estimate (mean of the gradient).

    beta2 : float ∈ [0, 1), optional (default=0.999)
        Decay rate for the second moment estimate (uncentered variance of the gradient).

    eps : float, optional (default=1e-08)
        Epsilon value for numerical stability.
        
    weight_decay : Real, optional (default=0)
        The weight decay term.

    Extended Description
    --------------------
    This optimizer implements the AdaMax variant of the Adam procedure described in the paper
      "Adam: A Method for Stochastic Optimization"
      Diederik P. Kingma and Jimmy Ba
    https://arxiv.org/abs/1412.6980
    """

    def __init__(
            self,
            params,
            *,
            learning_rate=0.002,
            beta1=0.9,
            beta2=0.999,
            eps=1e-08,
            weight_decay=0,
    ):
        assert 0 <= beta1 < 1
        assert 0 <= beta2 < 1
        self.params = params
        self.learning_rate = learning_rate
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps
        self.t = 0  # timestep
        self.m = []  # first moment
        self.u = []  # exponentially-weighted infinity norm
        for param in params:
            self.m.append(np.zeros_like(param.data))
            self.u.append(np.zeros_like(param.data))
        self.weight_decay = weight_decay

    def step(self):
        """ Perform one optimization step.

        This function should be called after accumulating gradients into the parameters of the model
        you wish to optimize via `backward()`. This will perform one step of the Adamax optimization
        algorithm put forward by Kingma and Ba.
        """
        self.t += 1
        for idx, param in enumerate(self.params):
            if param.grad is None:
                continue

            self.m[idx] = self.beta1 * self.m[idx] + (1 - self.beta1) * param.grad
            self.u[idx] = np.maximum(self.beta2 * self.u[idx], np.abs(param.grad))

            update = -self.weight_decay * param.data if param.ndim > 1 else 0
            tmp = self.u[idx] + self.eps
            update += -self.learning_rate / (1 - self.beta1**self.t) * self.m[idx] / tmp
            param.data += update
