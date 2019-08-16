import numpy as np


class Adadelta:
    """ Performs the Adadelta optimization procedure from Zeiler.

    Parameters
    ----------
    params : Iterable
        The parameters over which to optimize.

    rho : float ∈ [0, 1), optional (default=0.95)
        The decay rate in the running average.

    eps : float, optional (default=1e-06)
        Epsilon value to better condition the denominator.


    Extended Description
    --------------------
    This optimizer implements the Adadelta optimization procedure described in the paper
      "ADADELTA: An Adaptive Learning Rate Method"
      Matthew Zeiler
    https://arxiv.org/abs/1212.5701
    """

    def __init__(self, params, *, rho=0.95, eps=1e-06):
        self.params = params
        self.rho = rho
        self.eps = eps

        self.g = []
        self.dx = []
        for param in params:
            self.g.append(np.zeros_like(param.data))
            self.dx.append(np.zeros_like(param.data))

    def step(self):
        """ Perform one optimization step.

        This function should be called after accumulating gradients into the parameters of the model
        you wish to optimize via `backward()`. This will perform one step of the adadelta
        optimization algorithm put forward by Zeiler.
        """
        for idx, param in enumerate(self.params):
            grad = param.grad
            if grad is None:
                continue

            self.g[idx] = self.rho * self.g[idx] + (1 - self.rho) * grad ** 2  # step 4
            sqrt_dx = np.sqrt(self.dx[idx] + self.eps)
            sqrt_g = np.sqrt(self.g[idx] + self.eps)
                              
            dx = -sqrt_dx / sqrt_g * grad                                      # step 5
            self.dx[idx] = self.rho * self.dx[idx] + (1 - self.rho) * dx ** 2  # step 6
            param.data += dx                                                   # step 7
