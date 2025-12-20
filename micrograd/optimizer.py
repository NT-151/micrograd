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


def mean_squared_error(y_true, y_pred):
    total_loss = sum([(true - pred)**2 for true, pred in zip(y_true, y_pred)])

    return total_loss


def vae_loss(reconstruction, target, mu, log_var, beta=1.0):
    """
    Variational Autoencoder loss = Reconstruction Loss + beta * KL Divergence

    Args:
        reconstruction: Value or List of Value objects (reconstructed output)
        target: List of Value objects (original input)
        mu: List of Value objects (mean of latent distribution)
        log_var: List of Value objects (log variance of latent distribution)
        beta: Weight for KL divergence term (default 1.0)

    Returns:
        Total VAE loss as a Value object
    """
    # Ensure reconstruction is a list
    if not isinstance(reconstruction, list):
        reconstruction = [reconstruction]
    if not isinstance(target, list):
        target = [target]

    # Reconstruction loss (MSE)
    recon_loss = mean_squared_error(target, reconstruction)

    # KL divergence: -0.5 * sum(1 + log_var - mu^2 - exp(log_var))
    # This encourages the latent distribution to be close to N(0,1)
    kl_terms = []
    for mu_i, log_var_i in zip(mu, log_var):
        # KL term for one dimension: -0.5 * (1 + log_var - mu^2 - exp(log_var))
        kl_term = -0.5 * (Value(1.0) + log_var_i - mu_i**2 - log_var_i.exp())
        kl_terms.append(kl_term)

    kl_loss = sum(kl_terms) if kl_terms else Value(0.0)

    # Total loss
    total_loss = recon_loss + beta * kl_loss

    return total_loss
