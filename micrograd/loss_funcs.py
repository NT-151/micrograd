from micrograd.engine import Value


def mean_squared_error(y_true, y_pred):
    total_loss = sum([(true - pred)**2 for true, pred in zip(y_true, y_pred)])
    mean_loss = total_loss / len(y_true)

    return mean_loss


def vae_loss(reconstruction, target, mu, log_var, beta=1.0):

    if not isinstance(reconstruction, list):
        reconstruction = [reconstruction]
    if not isinstance(target, list):
        target = [target]

    # reconstruction loss
    recon_loss = mean_squared_error(target, reconstruction)

    # kl divergence = -0.5 * sum(1 + log_var - mu^2 - exp(log_var))
    kl_terms = []
    for mu_i, log_var_i in zip(mu, log_var):
        kl_term = -0.5 * (Value(1.0) + log_var_i - mu_i**2 - log_var_i.exp())
        kl_terms.append(kl_term)

    kl_loss = sum(kl_terms) / len(kl_terms) if kl_terms else Value(0.0)

    total_loss = recon_loss + beta * kl_loss

    return total_loss
