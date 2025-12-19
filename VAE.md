# Variational Autoencoders (VAEs): the idea, the ELBO, and the reparameterisation trick (with all the maths)

## 1) What the VAE paper is really about

When people hear “Variational Autoencoder,” they often think the paper introduced a new kind of autoencoder. But the main contribution is the **Stochastic Gradient Variational Bayes (SGVB)** method. SGVB is the practical algorithm that makes VAE training work at scale.

The problem SGVB tackles is Bayesian inference with latent variables. Once you add a hidden variable $(z)$, the math you want usually turns into an integral you cannot compute exactly.

---

## 2) The core problem: the marginal likelihood is intractable

A standard latent-variable model defines the joint distribution as:

$$
p_\theta(x,z) = p_\theta(z),p_\theta(x\mid z)
$$

What we want to maximise is the **marginal likelihood** (the “evidence”):

$$
p_\theta(x)=\int p_\theta(x,z),dz
$$

and often:

$$
\log p_\theta(x) = \log\int p_\theta(x,z),dz
$$

This integral over all $(z)$ is usually intractable (no closed form, or too expensive to approximate well).

People try methods like Monte Carlo EM or MCMC, but these can be slow and painful on large datasets. That is why we use **variational inference** as a faster approximation approach.

---

## 3) Variational inference: introduce (q\_\phi(z\mid x))

Instead of trying to compute the true posterior $(p_\theta(z\mid x))$ exactly, we introduce a simpler approximation:

$$
q_\phi(z\mid x) \approx p_\theta(z\mid x)
$$

We measure how close they are using **KL divergence**:

$$
D_{KL}\big(q_\phi(z\mid x),|,p_\theta(z\mid x)\big)
===================================================

\mathbb{E}*{q*\phi(z\mid x)}
\left[
\log \frac{q_\phi(z\mid x)}{p_\theta(z\mid x)}
\right]
\ge 0
$$

The fact that KL divergence is always non-negative is what creates the “lower bound” idea.

---

## 4) Deriving the ELBO (fully mapped out)

Start from the KL definition:

$$
D_{KL}\big(q_\phi(z\mid x),|,p_\theta(z\mid x)\big)
===================================================

\mathbb{E}*{q*\phi(z\mid x)}\left[\log q_\phi(z\mid x)-\log p_\theta(z\mid x)\right]
$$

Use Bayes’ rule:

$$
p_\theta(z\mid x)=\frac{p_\theta(x,z)}{p_\theta(x)}
$$

Take logs:

$$
\log p_\theta(z\mid x)=\log p_\theta(x,z)-\log p_\theta(x)
\quad
(\log(a/b)=\log a-\log b)
$$

Substitute:

$$
D_{KL}
======

\mathbb{E}*{q*\phi}\left[
\log q_\phi(z\mid x)
--------------------

\big(\log p_\theta(x,z)-\log p_\theta(x)\big)
\right]
$$

Distribute the minus sign:

$$
D_{KL}
======

\mathbb{E}*{q*\phi}\left[
\log q_\phi(z\mid x)-\log p_\theta(x,z)+\log p_\theta(x)
\right]
$$

Now use linearity of expectation:

$$
\mathbb{E}[A+B-C] = \mathbb{E}[A]+\mathbb{E}[B]-\mathbb{E}[C]
$$

Also note: $(\log p_\theta(x))$ does **not** depend on $(z)$. So under an expectation over $(z)$, it is a constant:

$$
\mathbb{E}*{q*\phi(z\mid x)}[\log p_\theta(x)] = \log p_\theta(x)
$$

So:

$$
D_{KL}
======

## \mathbb{E}*{q*\phi}\left[\log q_\phi(z\mid x)\right]

\mathbb{E}*{q*\phi}\left[\log p_\theta(x,z)\right]
+
\log p_\theta(x)
$$

Rearrange to solve for $(\log p_\theta(x))$:

$$
\log p_\theta(x)
================

D_{KL}\big(q_\phi(z\mid x),|,p_\theta(z\mid x)\big)
+
\mathbb{E}*{q*\phi}\left[\log p_\theta(x,z)\right]
--------------------------------------------------

\mathbb{E}*{q*\phi}\left[\log q_\phi(z\mid x)\right]
$$

Group the expectation terms and define the **ELBO**:

$$
\mathcal{L}(\theta,\phi;x)
==========================

\mathbb{E}*{q*\phi(z\mid x)}
\left[\log p_\theta(x,z)-\log q_\phi(z\mid x)\right]
$$

Then the key identity is:

$$
\log p_\theta(x)
================

\mathcal{L}(\theta,\phi;x)
+
D_{KL}\big(q_\phi(z\mid x),|,p_\theta(z\mid x)\big)
$$

Since (D\_{KL}\ge 0), we get:

$$
\mathcal{L}(\theta,\phi;x)\ \le\ \log p_\theta(x)
$$

That is why it is called the **Evidence Lower Bound (ELBO)**.

---

## 5) ELBO in the common VAE form

Expand the joint:

$$
p_\theta(x,z)=p_\theta(x\mid z),p_\theta(z)
$$

So:

$$
\log p_\theta(x,z)=\log p_\theta(x\mid z)+\log p_\theta(z)
$$

Plug into ELBO:

$$
\mathcal{L}(\theta,\phi;x)
==========================

\mathbb{E}*{q*\phi(z\mid x)}
\left[\log p_\theta(x\mid z)+\log p_\theta(z)-\log q_\phi(z\mid x)\right]
$$

Split:

$$
\mathcal{L}
===========

\mathbb{E}*{q*\phi}\left[\log p_\theta(x\mid z)\right]
+
\mathbb{E}*{q*\phi}\left[\log p_\theta(z)-\log q_\phi(z\mid x)\right]
$$

Recognise the KL term:

$$
D_{KL}\big(q_\phi(z\mid x),|,p_\theta(z)\big)
=============================================

\mathbb{E}*{q*\phi}\left[\log q_\phi(z\mid x)-\log p_\theta(z)\right]
$$

So:

$$
\mathbb{E}*{q*\phi}\left[\log p_\theta(z)-\log q_\phi(z\mid x)\right]
=====================================================================

* D_{KL}\big(q_\phi(z\mid x),|,p_\theta(z)\big)
$$

Final common form:

$$
\boxed{
\mathcal{L}(\theta,\phi;x)
==========================

## \mathbb{E}*{q*\phi(z\mid x)}\left[\log p_\theta(x\mid z)\right]

D_{KL}\big(q_\phi(z\mid x),|,p_\theta(z)\big)
}
$$

This matches the main intuition: maximising ELBO tries to (1) reconstruct data well and (2) keep the latent distribution close to the prior.

---

## 6) Gradients: why (\theta) is easier than (\phi)

### Gradient with respect to (\theta)

The expectation is over $(q_\phi(z\mid x))$, which depends on $(\phi)$, not $(\theta)$. So (under the same “move derivative inside the integral” idea):

$$
\nabla_\theta \mathcal{L}
=========================

\nabla_\theta \mathbb{E}*{q*\phi(z\mid x)}
\left[\log p_\theta(x,z)-\log q_\phi(z\mid x)\right]
$$

Since $(\log q_\phi(z\mid x))$ does not depend on $(\theta)$:

$$
\nabla_\theta \log q_\phi(z\mid x)=0
$$

So:

$$
\boxed{
\nabla_\theta \mathcal{L}
=========================

\mathbb{E}*{q*\phi(z\mid x)}
\left[\nabla_\theta \log p_\theta(x,z)\right]
}
$$

This is easy to estimate using Monte Carlo samples $(z\sim q_\phi(z\mid x))$.

### Gradient with respect to (\phi)

For $(\phi)$, the expectation itself depends on $(\phi)$:

$$
\mathcal{L}(\theta,\phi;x)=\mathbb{E}*{q*\phi(z\mid x)}[\cdot]
$$

So $(\nabla_\phi)$ affects both:

- the inside expression, and
- the sampling distribution $(q_\phi(z\mid x))$

That is where the “hard term” comes from.

---

## 7) The reparameterisation trick: the fix

The trick is to rewrite sampling so the randomness comes from a noise variable that does not depend on $(\phi)$.

Write:

- sample noise: $(e \sim p(e))$
- build $(z)$ as a deterministic function:

$$
z = g_\phi(x,e)
$$

Then:

$$
\mathbb{E}*{q*\phi(z\mid x)}[f(z)]
==================================

\mathbb{E}*{p(e)}[f(g*\phi(x,e))]
$$

Now the gradient becomes:

$$
\nabla_\phi,\mathbb{E}*{p(e)}[f(g*\phi(x,e))]
=============================================

\mathbb{E}*{p(e)}\left[\nabla*\phi f(g_\phi(x,e))\right]
$$

That makes Monte Carlo gradient estimates work cleanly.

---

## 8) The standard VAE Gaussian setup (and analytic KL)

A very common choice is:

$$
p(z)=\mathcal{N}(0,I)
$$

and

$$
q_\phi(z\mid x)=\mathcal{N}(\mu,\sigma)
$$

with $(\mu)$ and $(\sigma)$ coming from an encoder network:

$$
\mu,\sigma = f_\phi(x)
$$

In practice, we often parameterise with **log variance** for stability:

$$
\log\sigma^2 = \text{log_var}
\quad\Rightarrow\quad
\sigma = \exp(0.5,\text{log_var})
$$

### Reparameterisation (the exact sampling formula)

$$
\epsilon \sim \mathcal{N}(0,1)
$$

$$
\boxed{
z = \mu + \sigma \odot \epsilon
\quad\text{where}\quad
\sigma = \exp(0.5,\text{log_var})
}
$$

($(\odot)$ means element-wise multiply.)

### KL divergence in closed form

For diagonal Gaussians:

$$
q_\phi(z\mid x)=\mathcal{N}(\mu,\operatorname{diag}(\sigma^2)),
\quad
p(z)=\mathcal{N}(0,I)
$$

the KL is:

$$
\boxed{
D_{KL}\big(q_\phi(z\mid x),|,p(z)\big)
======================================

\frac{1}{2}\sum_{j=1}^d
\left(\mu_j^2+\sigma_j^2-\log\sigma_j^2-1\right)
}
$$

In $(\text{log_var})$ form $((\log\sigma_j^2 = \text{log_var}_j)$, $(\sigma_j^2=\exp(\text{log_var}_j)))$ this is commonly written as:

$$
\boxed{
D_{KL}\big(q_\phi(z\mid x),|,p(z)\big)
======================================

-\frac{1}{2}\sum_{j=1}^d
\left(1 + \text{log_var}_j - \mu_j^2 - \exp(\text{log_var}_j)\right)
}
$$

So the ELBO becomes:

$$
\mathcal{L}(\theta,\phi;x)
==========================

## \mathbb{E}*{q*\phi(z\mid x)}[\log p_\theta(x\mid z)]

D_{KL}\big(q_\phi(z\mid x),|,p(z)\big)
$$

The KL term is analytic now, so the only part you still estimate with sampling is the likelihood term.

---

## 9) Monte Carlo estimation, minibatches, and “single-sample”

To approximate the likelihood term:

$$
\mathbb{E}*{q*\phi(z\mid x)}[\log p_\theta(x\mid z)]
$$

sample (L) noises (e^{(1)},\dots,e^{(L)}\sim p(e)), build (z^{(l)}=g\_\phi(x,e^{(l)})), and use:

$$
\mathbb{E}*{q*\phi(z\mid x)}[\log p_\theta(x\mid z)]
\approx
\frac{1}{L}\sum_{l=1}^L \log p_\theta(x\mid z^{(l)})
$$

For a dataset with $(N)$ points, using a minibatch of size $(M)$:

$$
\sum_{i=1}^N \mathcal{L}(\theta,\phi;x_i)
\approx
\frac{N}{M}\sum_{i=1}^M \mathcal{L}(\theta,\phi;x_i)
$$

A common practical choice is:

- **single-sample Monte Carlo**: $(L=1)$ sample per datapoint

---

## 10) VAEs as “autoencoder + probability”

A normal autoencoder encodes $(x)$ into a single vector and decodes it back.

A VAE is different:

- the encoder outputs **distribution parameters**, not just one vector
- you sample $(z)$, then decode
- training uses a reconstruction loss plus a KL term that pushes the latent space toward the prior

A typical training objective is the negative ELBO, often written like:

$$
\boxed{
\text{total loss}
=================

\text{reconstruction loss}
+
\beta;D_{KL}\big(q_\phi(z\mid x),|,p(z)\big)
}
$$

where $(\beta)$ is a knob that controls the strength of the KL term.

---

## 11) Implementation notes (Micrograd-style, scalar values)

To understand the ideas deeply, you described implementing a VAE using **micrograd**, which uses scalar `Value` objects instead of tensors.

Your learning path was:

1. Build a standard autoencoder using two MLPs (encoder + decoder).
2. Wrap both into a class where calling the class returns the reconstruction.
3. Test it on a tiny binary dataset to confirm it learns to reconstruct.
4. Convert it into a VAE by changing what the encoder outputs.

Key VAE change:

- The encoder must output $(\mu)$ and $(\text{log_var})$ (or $(\sigma)$) for each latent dimension.
- If you store them in one vector, you “double” the latent output size and then split:

$$
\text{encoder}(x) \to [\mu ;;; \text{log_var}]
$$

Then sample:

$$
\epsilon \sim \mathcal{N}(0,1),\quad
\sigma = \exp(0.5,\text{log_var}),\quad
z = \mu + \sigma \odot \epsilon
$$

You also noted a design choice: you tried to keep $$(\epsilon)$$ out of the parameter set by creating it as a standalone `Value` with no trainable parameters.

Your forward pass returns three things:

- reconstructed $(x)$
- $(\mu)$
- $(\text{log_var})$

because your loss needs all three:

- reconstruction loss compares $(x)$ and $(\hat{x})$ (MSE or BCE)
- KL uses $(\mu)$ and $(\text{log_var})$ against the prior $(p(z)=\mathcal{N}(0,I))$

You also said this was confusing at first because the paper math did not feel “mapped” to the implementation details, especially:

- doubling and splitting into $(\mu)$ and $(\text{log_var})$
- how the reparameterisation trick shows up in code

---

## 12) A few unclear spots (and clearer versions)

- **“Big enough minibatch” is vague.**
  Clearer version: pick $(M)$ so training is stable (loss and gradients do not jump wildly), then use $(L=1)$ per datapoint in practice.

- **The “constant inside expectation” idea can be stated cleanly.**
  Clearer version: if a term does not depend on (z), then
  $$
  \mathbb{E}*{q*\phi(z\mid x)}[c] = c
  $$
  so it can be pulled out.

---

### Quick credit note (from the original text)

You referenced sources like “Understanding Variational Autoencoders (VAEs) - DeepBean,” and also pointed to blog/tutorial explanations for parts of the reparameterisation discussion. The post above keeps your core chain of ideas, but lays out every step in the maths so the link between equations and implementation is clearer.
