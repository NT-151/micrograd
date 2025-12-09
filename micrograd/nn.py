import random
import math
from micrograd.engine import Value


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
        tied_weights = kwargs.get('tied_weights', None)
        self.w = tied_weights if tied_weights is not None else [
            Value(random.uniform(-1, 1)) for _ in range(nin)]
        # self.b = Value(random.uniform(-1, 1))
        self.b = Value(0)
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
    def __init__(self, nin, nout, tied_to_layer=None, **kwargs):
        if tied_to_layer is None:
            # Standard Layer initialization
            self.neurons = [Neuron(nin, **kwargs) for _ in range(nout)]
        else:
            # Tied Layer initialization
            # The weights for this layer are the transpose of the tied_to_layer's weights
            # This requires careful construction.
            # Number of inputs for this layer = number of outputs of the tied layer
            # Number of outputs for this layer = number of inputs of the tied layer
            self.tied_to_layer = tied_to_layer
            self.neurons = [Neuron(nin, **kwargs) for _ in range(nout)]

    def __call__(self, x):
        if hasattr(self, 'tied_to_layer'):
            # The weights are conceptually transposed.
            # So, the output of a neuron is sum(w_ji * x_i), which means summing over the neurons of the previous layer.
            # This is hard to do cleanly with the current structure.
            # The simpler approach is to loop manually.
            out = []
            for j in range(len(self.neurons)):
                # The j-th neuron of this layer uses the j-th weight of every neuron in the tied layer.
                # For each output neuron (j), sum the weighted inputs.
                # The weight connecting input `i` to output `j` is the same as the weight connecting
                # input `j` of the encoder layer to output `i`.
                act = sum(self.tied_to_layer.neurons[i].w[j] * x[i] for i in range(len(x))) + self.neurons[j].b if isinstance(
                    x, list) else self.tied_to_layer.neurons[0].w[j] * x + self.neurons[j].b

                # Apply activation
                if self.neurons[j].activate and self.neurons[j].nonlin is False:
                    act = self.neurons[j].activate(act)
                else:
                    act = act.relu() if self.neurons[j].nonlin else act
                out.append(act)
            return out[0] if len(out) == 1 else out
        else:
            # Standard layer behavior
            out = [n(x) for n in self.neurons]
            return out

    def parameters(self):
        # In a tied layer, the weights are shared, but the biases are not.
        if hasattr(self, 'tied_to_layer'):
            return [n.b for n in self.neurons]
        else:
            return [p for n in self.neurons for p in n.parameters()]

    def __repr__(self):
        return f"Layer of [{', '.join(str(n) for n in self.neurons)}]"


class MLP(Module):
    def __init__(self, nin, nouts, tied_weights_from=None, **kwargs):
        sz = [nin] + nouts
        self.layers = []
        if tied_weights_from is None:
            # Standard MLP initialization
            self.layers = [Layer(
                sz[i], sz[i+1], nonlin=i != len(nouts)-1, **kwargs) for i in range(len(nouts))]
        else:
            # Tied-weight MLP initialization
            tied_layers = list(reversed(tied_weights_from))
            for i in range(len(nouts)):
                # Pass the encoder's layer directly to the decoder's layer.
                # The decoder layer will use the encoder's weights.
                self.layers.append(Layer(
                    sz[i], sz[i+1], tied_to_layer=tied_layers[i], nonlin=i != len(nouts)-1, **kwargs))

    def __call__(self, x):
        for layer in self.layers:
            x = layer(x)
        return x

    def parameters(self):
        return [p for layer in self.layers for p in layer.parameters()]

    def __repr__(self):
        return f"MLP of [{', '.join(str(layer) for layer in self.layers)}]"


class AutoEncoder(Module):
    def __init__(self, in_embeds=1, hidden_layers=[], latent_dim=1, act_func=None, tied=False):
        self.latent_dim = latent_dim
        self.act_func = act_func
        self.encoder = MLP(in_embeds, hidden_layers + [latent_dim])

        # Create decoder, passing encoder layers for tied weights
        if tied:
            self.decoder = MLP(latent_dim, list(reversed(
                hidden_layers)) + [in_embeds], tied_weights_from=self.encoder.layers, activate=act_func)
        else:
            self.decoder = MLP(latent_dim, list(
                reversed(hidden_layers)) + [in_embeds], activate=act_func)

    def __call__(self, x):
        compressed = self.encoder(x)
        out = self.decoder(compressed)
        return out

    def encode(self, x):
        encoded = self.encoder(x)
        return encoded

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
    """
    Simple Variational Autoencoder implementation.
    The encoder outputs mean and log-variance for each latent dimension.
    Uses reparameterization trick to sample from the latent distribution.
    """

    def __init__(self, in_embeds=1, hidden_layers=[], latent_dim=1, act_func=None, tied=False):
        self.latent_dim = latent_dim
        self.act_func = act_func

        # Encoder outputs 2 * latent_dim: mean and log-variance for each dimension
        # Last layer outputs 2*latent_dim (no activation on this layer)
        self.encoder = MLP(in_embeds, hidden_layers + [2 * latent_dim])

        # Decoder takes latent_dim as input
        if tied:
            # For tied weights, we'd need to handle the 2*latent_dim -> latent_dim transition
            # For simplicity, we'll skip tied weights in VAE for now
            self.decoder = MLP(latent_dim, list(
                reversed(hidden_layers)) + [in_embeds], activate=act_func)
        else:
            self.decoder = MLP(latent_dim, list(
                reversed(hidden_layers)) + [in_embeds], activate=act_func)

    def encode(self, x):
        """Encode input to mean and log-variance"""
        encoded = self.encoder(x)

        # Ensure encoded is a list
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
        epsilon = [Value(random.gauss(0, 1)) for _ in range(len(mu))]

        # Compute sigma = exp(0.5 * log_var) more efficiently
        # sigma = exp(0.5 * log_var) = sqrt(exp(log_var))
        sigma = [(log_var_i * 0.5).exp() for log_var_i in log_var]

        # z = mu + sigma * epsilon
        z = [mu_i + sigma_i * eps_i for mu_i,
             sigma_i, eps_i in zip(mu, sigma, epsilon)]

        return z

    def decode(self, z):
        """Decode latent sample to reconstruction"""
        return self.decoder(z)

    def __call__(self, x):
        """Forward pass: encode, sample, decode"""
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
