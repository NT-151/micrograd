# Variational AutoEncoders

The paper Auto-Encoding Variational Bayes introduced the SGVB algorithm, a way to perform variational inference. It also produced Variational AutoEncoders, which is a popular way to perform variational inference with neural networks.

## Marginal likelihood

A standard latent-variable model defines the joint distribution:

$$
p_\theta(z,x) = p_\theta(z)\,p_\theta(x\mid z)
$$

We want to maximise the marginal likelihood of the observed dataset:

$$
p_\theta(x)=\int_z p_\theta(z)\,p_\theta(x\mid z)\;dz
$$

This integral over all $z$ is usually intractable (no closed form, or too expensive to approximate well).[1]

People try methods like Monte Carlo EM or MCMC, but these can be slow and costly on large datasets. That is why we use **variational inference** as a faster approximation approach.

## Variational inference

Variational Inference is trying to infer conditional distributions over the latent variables given the observations, which can be modelled using Bayes Theorem:

$$
p_{\theta }
(z|x)=\frac{p_{\theta }(z)\;p_\theta(x|z)}{p_{\theta }(x)}  [2]
$$

Because $p_\theta(x)$ is intractable, the posterior $p_\theta(z|x)$ is also intractable.

Instead of trying to compute the true posterior $p_\theta(z\mid x)$, we introduce a simpler approximation:

$$
q_\phi(z\mid x) \approx p_\theta(z\mid x)
$$

We measure how close they are using **KL divergence**:

$$
D_{KL}\big(q_\phi(z\mid x)||p_\theta(z\mid x)\big)=
\mathbb{E}_{q_\phi(z\mid x)}
\left[
\log  \frac{q_\phi(z\mid x)}{p_\theta(z\mid x)}
\right]
\ge  0
$$

With the goal of minimisation.

## Deriving ELBO

Start from the KL definition:

$$
D_{KL}\!\left(q_\phi(z\mid x)||p_\theta(z\mid x)\right)
=\mathbb{E}_{q_\phi(z\mid x)}\!\left[\log q_\phi(z\mid x)-\log p_\theta(z\mid x)\right]
$$

Use Bayes’ rule:

$$
\log p_\theta(z\mid x)=\log p_\theta(x,z)-\log p_\theta(x)
$$

Substitute:

$$
\begin{aligned}
D_{KL}\!\left(q_\phi(z\mid x)||p_\theta(z\mid x)\right)
&=\mathbb{E}_{q_\phi}\!\left[\log q_\phi(z\mid x)-\left(\log p_\theta(x,z)-\log p_\theta(x)\right)\right] \\
&=\mathbb{E}_{q_\phi}\!\left[\log q_\phi(z\mid x)-\log p_\theta(x,z)+\log p_\theta(x)\right]
\end{aligned}
$$

Apply linearity of expectations:

$$
\mathbb{E}[A-B+C] = \mathbb{E}[A]-\mathbb{E}[B]+\mathbb{E}[C]
$$

$$
\begin{aligned}
D_{KL}\!\left(q_\phi(z\mid x)||p_\theta(z\mid x)\right)
&=\mathbb{E}_{q_\phi}\!\left[\log q_\phi(z\mid x)\right]
-\mathbb{E}_{q_\phi}\!\left[\log p_\theta(x,z)\right]
+\mathbb{E}_{q_\phi}\!\left[\log p_\theta(x)\right]
\end{aligned}
$$

Since $\log p_\theta(x)$ does not depend on $z$ it is a constant w.r.t the expectation:

$$
\mathbb{E}_{q_\phi}\!\left[\log p_\theta(x)\right]=\log p_\theta(x)
$$

So:

$$
D_{KL}\!\left(q_\phi(z\mid x)||p_\theta(z\mid x)\right)=
\mathbb{E}_{q_\phi}\!\left[\log q_\phi(z\mid x)\right]
-\mathbb{E}_{q_\phi}\!\left[\log p_\theta(x,z)\right]
+\log p_\theta(x)
$$

Rearrange to isolate $\log p_\theta(x)$:

$$
\log p_\theta(x)=
\mathbb{E}_{q_\phi}\!\left[\log p_\theta(x,z)\right]
-\mathbb{E}_{q_\phi}\!\left[\log q_\phi(z\mid x)\right]
+
D_{KL}\!\left(q_\phi(z\mid x)||p_\theta(z\mid x)\right)
$$

Group the expectation terms and define the **ELBO**:

$$
\mathcal{L}(\theta,\phi;x)=
\mathbb{E}_{q_\phi(z\mid x)}
\left[\log p_\theta(x,z)-\log q_\phi(z\mid x)\right]
$$

Then the key identity is:

$$
\log p_\theta(x)=
\mathcal{L}(\theta,\phi;x)+
D_{KL}\big(q_\phi(z\mid x)||p_\theta(z\mid x)\big)
$$

Since $D_{KL}\ge 0$, we get:

$$
\log p_\theta(x) \ge
\mathcal{L}(\theta,\phi;x)
$$

That is why it is called the **Evidence Lower Bound (ELBO)**.

## ELBO in VAE form

Expand the joint:

$$
p_\theta(x,z)=p_\theta(z)\,p_\theta(x\mid z)
$$

So:

$$
\log p_\theta(x,z)=\log p_\theta(x\mid z)+\log p_\theta(z)
$$

Plug into ELBO:

$$
\mathcal{L}(\theta,\phi;x)=
\mathbb{E}_{q_\phi(z\mid x)}
\left[\log p_\theta(x\mid z)+\log p_\theta(z)-\log q_\phi(z\mid x)\right]
$$

Split:

$$
\mathcal{L}=
\mathbb{E}_{q_\phi}\left[\log p_\theta(x\mid z)\right] +
\mathbb{E}_{q_\phi}\left[\log p_\theta(z)-\log q_\phi(z\mid x)\right]
$$

Recognise the KL term:

$$
D_{KL}\big(q_\phi(z\mid x)||p_\theta(z)\big)=
\mathbb{E}_{q_\phi}\left[\log q_\phi(z\mid x)-\log p_\theta(z)\right]
$$

So:

$$
\mathbb{E}_{q_\phi}\left[\log p_\theta(z)-\log q_\phi(z\mid x)\right]=
-D_{KL}\big(q_\phi(z\mid x)||p_\theta(z)\big)
$$

Final common form:

$$
\boxed{
\mathcal{L}(\theta,\phi;x)=
\mathbb{E}_{q_\phi(z\mid x)}\left[\log p_\theta(x\mid z)\right]
-D_{KL}\big(q_\phi(z\mid x)||p_\theta(z)\big)
}
$$

This matches the main intuition: maximising ELBO tries to reconstruct data well and also keep the latent distribution close to the prior.

## Why $\theta$ is easier than $\phi$

### Gradient with respect to $\theta$

The expectation is over $q_\phi(z\mid x)$, which depends on $\phi$, not $\theta$. So (under the same “move derivative inside the integral” idea):

$$
\nabla_\theta  \mathcal{L}=
\nabla_\theta  \mathbb{E}_{q_\phi(z\mid x)}
\left[\log p_\theta(x,z)-\log q_\phi(z\mid x)\right]
$$

Since $\log q_\phi(z\mid x)$ does not depend on $\theta$:

$$
\nabla_\theta  \log q_\phi(z\mid x)=0
$$

So:

$$
\boxed{
\nabla_\theta  \mathcal{L}=
\mathbb{E}_{q_\phi(z\mid x)}
\left[\nabla_\theta  \log p_\theta(x,z)\right]
}
$$

This is easy to estimate using Monte Carlo samples $z\sim q_\phi(z\mid x)$.

### Gradient with respect to $\phi$

For $\phi$, the expectation itself depends on $\phi$:

$$
\mathcal{L}(\theta,\phi;x)=\mathbb{E}_{q_\phi(z\mid x)}[\cdot]
$$

So $\nabla_\phi$ affects both:

- the inside expression, and

- the sampling distribution $q_\phi(z\mid x)$

That is where the “hard term” comes from.

## Reparameterisation trick

The trick is to rewrite sampling so the randomness comes from a noise variable that does not depend on $\phi$.

- sample noise: $\epsilon \sim p(\epsilon)$

- build $z$ as a deterministic function:

$$
z = g_\phi(x,\epsilon)
$$

Then:

$$
\mathbb{E}_{q_\phi(z\mid x)}[f(z)]=
\mathbb{E}_{p(\epsilon)}[f(g_\phi(x,\epsilon))]
$$

Assuming regularity so we can interchange gradient and expectation:

$$
\nabla_\phi\mathbb{E}_{p(\epsilon)}[f(g_\phi(x,\epsilon))]=
\mathbb{E}_{p(\epsilon)}\left[\nabla_\phi f(g_\phi(x,\epsilon))\right]
$$

Which makes Monte Carlo gradient estimates work cleanly.

## VAE Gaussian setup and analytic KL

A very common choice is:

$$
p_\theta(z)=\mathcal{N}(0,I)
$$

and

$$
q_\phi(z\mid x)=\mathcal{N}(\mu(x),\text{diag}(\sigma^2(x)))
$$

with $\mu$ and $\sigma$ coming from an encoder network:

$$
\mu,\sigma = f_\phi(x)
$$

In practice, we often parameterise with **log variance** for stability:

$$
\log\sigma^2 = {\log\text {var}}
\quad\Rightarrow\quad
\sigma = \exp(\frac{1}{2}\;log\_var)
$$

### Reparameterisation (the exact sampling formula)

$$
\epsilon  \sim  \mathcal{N}(0,1)
$$

$$
\boxed{
z = \mu + \sigma  \odot  \epsilon
\quad\text{where}\quad
\sigma = \exp(\frac{1}{2}\;\log\_var)
}
$$

($\odot$ means element-wise multiply.)

### KL divergence in closed form

For diagonal Gaussians:

$$
q_\phi(z\mid x)=\mathcal{N}(\mu(x),\text{diag}(\sigma^2(x))),
\quad
p_\theta(z)=\mathcal{N}(0,I)
$$

$$
\boxed{
D_{KL}\big(q_\phi(z\mid x)||p_\theta(z)\big)=
\frac{1}{2}\sum_{j=1}^d
\left(\mu_j^2+\sigma_j^2-\log\sigma_j^2-1\right)
}
$$

In ${\log \text{var}}$ form $(\log\sigma_j^2 = {\log\text {var}}_j$, $\sigma_j^2=\exp({\log\text {var}}_j))$ this is commonly written as:

$$
\boxed{
D_{KL}\!\left(q_\phi(z\mid x)||p_\theta(z)\right)=
-\frac{1}{2}\sum_{j=1}^{d}
\left(1+{\log\text {var}}_j-\mu_j^2-\exp\!\left({\log\text {var}}_j\right)\right)
}
$$

So the ELBO becomes:

$$
\mathcal{L}(\theta,\phi; x) =
\mathbb{E}_{q_{\phi}(z\mid x)}\!\left[\log p_\theta(x\mid z)\right] -
D_{KL}\!\left(q_{\phi}(z\mid x)||p_\theta(z)\right)
$$

The KL term is analytic now, so the only part you still estimate with sampling is the likelihood term.

## Monte Carlo estimation, minibatches, and “single-sample”

To approximate the likelihood term:

$$
\mathbb{E}_{q_\phi(z\mid x)}[\log p_\theta(x\mid z)]
$$

sample $L$ noises $\epsilon^{(1)},\dots,\epsilon^{(L)}\sim p(\epsilon)$, build $z^{(l)}=g\_\phi(x,\epsilon^{(l)})$, and use:

$$
\mathbb{E}_{q_\phi(z\mid x)}[\log p_\theta(x\mid z)]
\approx
\frac{1}{L}\sum_{l=1}^L \log p_\theta(x\mid z^{(l)})
$$

For a dataset with $N$ points, using a minibatch of size $M$:

$$
\sum_{i=1}^N \mathcal{L}(\theta,\phi;x_i)
\approx
\frac{N}{M}\sum_{i=1}^M \mathcal{L}(\theta,\phi;x_i)
$$

A common practical choice is:

- **single-sample Monte Carlo**: $L=1$ sample per datapoint

## AutoEncoder + Probability

A normal autoencoder encodes $x$ into a single vector and decodes it back.

A VAE is different:

- the encoder outputs **distribution parameters**, not just one vector

- you sample $z$, then decode

- training uses a reconstruction loss plus a KL term that pushes the latent space toward the prior

A typical training objective is the negative ELBO, often written like:

$$
\boxed{
\begin{aligned}
&\text{total loss} =
\text{reconstruction loss} + \beta \cdot D_{KL}\big(q_\phi(z \mid x)|| p_\theta(z)\big)
\end{aligned}
}
$$

where $\beta$ is a knob that controls the strength of the KL term.

## Implementation notes

I wanted to learn more about the architecture behind VAE's, I implemented it using **micrograd**, by Andrej Karpathy [3], which uses scalar `Value` objects instead of tensors.

I first implemented an AutoEncoder using MLPs. The architecture consists of an encoder that compresses the input and a symmetrical decoder that mirrors the encoder's structure to reconstruct the data.

I found the main differences were:

- The encoder must output $\mu$ and ${\log\text {var}}$ (or $\sigma)$ for each latent variable, so I doubled the size of compressed layer and split it into two heads

- Reparameterisation trick to sample latent vector which included $\epsilon$, treated as a constant during backprop

- Loss function required reconstructed input as well as $\mu$ and $\sigma$

Code can be found in demo.ipynb

---

[1] https://arxiv.org/pdf/1312.6114
[2] https://www.cs.cmu.edu/~epxing/Class/10708-15/notes/10708_scribe_lecture13.pdf
[3] https://github.com/karpathy/micrograd
