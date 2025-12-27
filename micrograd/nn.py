import random
import math
from engine import Value


class Module:

    def zero_grad(self):
        for p in self.parameters():
            p.grad = 0

    def parameters(self):
        return []

    def layers(self):
        return []

    def summary(self):
        return f"{len(self.layers())} layers, {len(self.parameters())} parameters"


class Neuron(Module):

    # I want to introduce weight sharing, which means I need to be able to
    # initialise a neuron with pre defined weights, but leave the bias?

    def __init__(self, nin, nonlin=True, **kwargs):
        k = math.sqrt(2 / nin)

        self.w = [Value(random.uniform(-k, k)) for _ in range(nin)]
        self.b = Value(random.uniform(-k, k))
        self.nonlin = nonlin
        self.activate = kwargs.get('activate', None)

    def __call__(self, x):
        if isinstance(x, (Value, float, int)):
            # This is for a single input, likely at the start of a layer
            act = (self.w[0] * x) + self.b
        else:
            act = sum((wi*xi for wi, xi in zip(self.w, x)), self.b)
        if self.activate and self.nonlin == False:
            return self.activate(act)
        else:
            return act.leaky_relu() if self.nonlin else act

    def parameters(self):
        return self.w + [self.b] if isinstance(self.w[0], Value) else [p for w_list in self.w for p in w_list] + [self.b]

    def __repr__(self):
        return f"{'ReLU' if self.nonlin else '{self.activate}'}Neuron({len(self.w)})"


class Layer(Module):
    def __init__(self, nin, nout, **kwargs):
        self.neurons = [Neuron(nin, **kwargs) for _ in range(nout)]

    def __call__(self, x):
        out = [n(x) for n in self.neurons]
        return out

    def parameters(self):
        return [p for n in self.neurons for p in n.parameters()]

    def __repr__(self):
        return f"Layer of [{', '.join(str(n) for n in self.neurons)}]"


class MLP(Module):
    def __init__(self, nin, nouts, **kwargs):
        sz = [nin] + nouts
        self.layers = []
        if tied_weights_from is None:
            # Standard MLP initialization
            self.layers = [Layer(
                sz[i], sz[i+1], nonlin=i != len(nouts)-1, **kwargs) for i in range(len(nouts))]

    def __call__(self, x):
        for layer in self.layers:
            x = layer(x)
        return x

    def parameters(self):
        return [p for layer in self.layers for p in layer.parameters()]

    def __repr__(self):
        return f"MLP of [{', '.join(str(layer) for layer in self.layers)}]"


class AutoEncoder(Module):
    def __init__(self, in_embeds=1, hidden_layers=[], latent_dim=1, act_func=None):
        self.latent_dim = latent_dim
        self.act_func = act_func
        self.encoder = MLP(in_embeds, hidden_layers + [latent_dim])
        self.decoder = MLP(latent_dim, list(
            reversed(hidden_layers)) + [in_embeds], activate=act_func)

    def __call__(self, x):
        compressed = self.encoder(x)
        out = self.decoder(compressed)
        return out

    def encode(self, x):
        encoded = self.encoder(x)
        return encoded

    def decode(self, x):
        decoded = self.decoder(x)
        return decoded

    def parameters(self):
        return self.encoder.parameters() + self.decoder.parameters()

    def layers(self):
        return self.encoder.layers + self.decoder.layers

    def pretty(self):
        if self.act_func != None:
            hey = str(self.act_func)
            return hey.split()[1][6:]
        else:
            return "no function"

    def __repr__(self):
        return f"encoder has {len(self.encoder.layers)}, decoder has {len(self.decoder.layers)}, latent dim is {self.latent_dim} activated with {self.pretty()}"


class VariationalAutoEncoder(Module):

    def __init__(self, in_embeds=1, hidden_layers=[], latent_dim=1, act_func=None):
        self.latent_dim = latent_dim
        self.act_func = act_func

        # encoder ouputs mean and log-variance for each dimension
        self.encoder = MLP(in_embeds, hidden_layers + [2 * latent_dim])
        self.decoder = MLP(latent_dim, list(
            reversed(hidden_layers)) + [in_embeds], activate=act_func)

    def encode(self, x):
        encoded = self.encoder(x)

        if not isinstance(encoded, list):
            encoded = [encoded]

        # Split the output into mean and log_var
        # encoded should be a list of 2*latent_dim values
        if len(encoded) != 2 * self.latent_dim:
            raise ValueError(
                f"Encoder output dimension {len(encoded)} doesn't match expected 2*latent_dim={2*self.latent_dim}")

        mu = encoded[:self.latent_dim]
        log_var = encoded[self.latent_dim:]

        return mu, log_var

    def reparameterize(self, mu, log_var):
        """
        Reparameterization trick: z = mu + sigma * epsilon
        where epsilon ~ N(0,1) and sigma = exp(0.5 * log_var)

        Note: epsilon is sampled and treated as a constant during backprop
        """
        # Sample epsilon from standard normal (treated as constant in backprop)
        # Using Value.constant() ensures gradients don't flow to epsilon
        epsilon = [Value.constant(random.gauss(0, 1)) for _ in range(len(mu))]

        # Compute sigma = exp(0.5 * log_var) more efficiently
        # sigma = exp(0.5 * log_var) = sqrt(exp(log_var))
        sigma = [(log_var_i * 0.5).exp() for log_var_i in log_var]

        # z = mu + sigma * epsilon
        z = [mu_i + sigma_i * eps_i for mu_i,
             sigma_i, eps_i in zip(mu, sigma, epsilon)]

        return z

    def decode(self, z):
        return self.decoder(z)

    def __call__(self, x):
        mu, log_var = self.encode(x)
        z = self.reparameterize(mu, log_var)
        reconstruction = self.decode(z)
        return reconstruction, mu, log_var

    def parameters(self):
        return self.encoder.parameters() + self.decoder.parameters()

    def layers(self):
        return self.encoder.layers + self.decoder.layers

    def __repr__(self):
        return f"VAE(encoder: {len(self.encoder.layers)} layers, decoder: {len(self.decoder.layers)} layers, latent_dim: {self.latent_dim})"


auto = AutoEncoder(
    in_embeds=784,  # input dimension
    hidden_layers=[312, 128, 64],  # len = number of layers, i = size of layer
    latent_dim=8,  # compressed layer dimensions
    act_func=Value.sigmoid,  # activation function for final decoder layer
)

print(len(auto.parameters()))
