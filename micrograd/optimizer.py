from micrograd.engine import Value


class Optimizer:
    """Base class for optimizers"""

    def __init__(self, parameters):
        self.parameters = parameters

    def zero_grad(self):
        for p in self.parameters:
            p.grad = 0

    def step(self):
        """Take a step of gradient descent"""

        raise NotImplementedError


class SGD(Optimizer):
    """Stochastic Gradient Descent optimizer"""

    def __init__(self, parameters, learning_rate=0.01):
        super().__init__(parameters)
        self.learning_rate = learning_rate

    def step(self):
        for p in self.parameters:
            p.data -= self.learning_rate * p.grad
