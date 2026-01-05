from engine import Value


def mean_squared_error(target, pred):
    loss = ((pred - target) ** 2).mean()

    return loss


def vae_loss(recon, target, mu, log_var, beta=0.002):
    # reconstruction loss (use mean to keep scales sane)
    recon_loss = mean_squared_error(target, recon)

    # kl per element
    kl_elem = (Value(1.0) + log_var - (mu * mu) - log_var.exp())

    # sum over latent dim; then mean over batch if batched
    if kl_elem.data.ndim == 1:
        kl = kl_elem.sum() * (-0.5)
    else:
        kl = kl_elem.sum(axis=1).mean() * (-0.5)

    total_loss = recon_loss + beta * kl

    # print("recon:", float(recon_loss.data), "kl:", float(kl.data), "total:", float(total_loss.data))
    return total_loss
