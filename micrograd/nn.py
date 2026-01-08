import tracemalloc
import os
import linecache
import time
from engine import Value
import numpy as np


class Module:

    def zero_grad(self):
        for p in self.parameters():
            p.grad = np.zeros_like(p.data)

    def parameters(self):
        return []


class Linear(Module):
    def __init__(self, nin, nout):
        k = np.sqrt(2.0 / nin)
        self.W = Value(np.random.uniform(-k, k, (nin, nout)))
        self.b = Value(np.full((nout,), 0.1))
        print(self.W.data.shape, self.b.data.shape, "this is in the weights initialisation")

    def __call__(self, x):
        return (x @ self.W) + self.b

    def parameters(self):
        return [self.W, self.b]


class MLP(Module):
    def __init__(self, nin, nouts, act_func=None):
        sz = [nin] + nouts
        self.layers = [Linear(sz[i], sz[i+1]) for i in range(len(nouts))]
        self.act_func = act_func

    def __call__(self, x):
        for i, layer in enumerate(self.layers):
            x = layer(x)
            if i != len(self.layers)-1:
                x = x.relu()
            else:
                if self.act_func is not None:
                    x = self.act_func(x)
        return x

    def parameters(self):
        return [p for layer in self.layers for p in layer.parameters()]


class AutoEncoder(Module):
    def __init__(self, in_embeds=1, hidden_layers=[], latent_dim=1, act_func=None):
        self.latent_dim = latent_dim
        self.act_func = act_func
        self.encoder = MLP(in_embeds, hidden_layers + [latent_dim])
        self.decoder = MLP(latent_dim, list(
            reversed(hidden_layers)) + [in_embeds], act_func=act_func)

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

    def __repr__(self):
        return f"encoder has {len(self.encoder.layers)}, decoder has {len(self.decoder.layers)}, latent dim is {self.latent_dim}"


class VariationalAutoEncoder(Module):
    def __init__(self, in_embeds, hidden_layers, latent_dim, act_func=None):
        self.latent_dim = latent_dim
        self.act_func = act_func

        # encoder outputs 2*latent_dim
        self.encoder = MLP(in_embeds, hidden_layers + [2 * latent_dim])
        self.decoder = MLP(latent_dim, list(
            reversed(hidden_layers)) + [in_embeds], act_func=act_func)

    def encode(self, x: Value):
        h = self.encoder(x)  # shape (B, 2d) or (2d,)

        d = self.latent_dim
        if h.data.ndim == 1:
            mu = h[:d]
            log_var = h[d:2*d]
        else:
            mu = h[:, :d]
            log_var = h[:, d:2*d]
        return mu, log_var

    def reparameterize(self, mu: Value, log_var: Value):
        # * inside randn() allows dynamic generation of numpy array
        eps = Value.constant(np.random.randn(*mu.data.shape))
        sigma = (log_var * 0.5).exp()
        z = mu + sigma * eps
        return z

    def decode(self, z: Value):
        return self.decoder(z)

    def __call__(self, x: Value):
        mu, log_var = self.encode(x)
        z = self.reparameterize(mu, log_var)
        recon = self.decode(z)
        return recon, mu, log_var

    def parameters(self):
        return self.encoder.parameters() + self.decoder.parameters()


pred = MLP(5, [1])


# start_time = time.time()
x = Value([
    [1.0, 2.0, 3.0, 0.0, 0.2],
    [0.5, 1.0, 3.0, -2.0, 0.0],
    [0.0, 0.0, -4.0, 2.0, 3.0],
    [0.0, 0.0, 0.0, -2.0, 2.5],
    [0.0, 0.0, 3.0, 4.0, -1.0],
])

truth = Value([1.0, 0.0, 1.0, 1.0, 1.0])

forward = pred(x)

loss = ((forward - truth)**2).mean()

loss.backward()

def backward(self):
    # topological order all of the children in the graph
    topo = []
    visited = set()

    def build_topo(v):
        if len(visited) == 10:
            return
        if v not in visited:
            visited.add(v)
            for child in v._prev:
                build_topo(child)
            if len(v.grad.shape) > 1:
                print(v.grad[0])
    build_topo(self)


backward(loss)


# print("--- %s seconds ---" % (time.time() - start_time))
 

def display_top(snapshot, key_type='lineno', limit=10):
    snapshot = snapshot.filter_traces((
        tracemalloc.Filter(False, "<frozen importlib._bootstrap>"),
        tracemalloc.Filter(False, "<unknown>"),
    ))
    top_stats = snapshot.statistics(key_type)

    print("Top %s lines" % limit)
    for index, stat in enumerate(top_stats[:limit], 1):
        frame = stat.traceback[0]
        print("#%s: %s:%s: %.1f KiB"
              % (index, frame.filename, frame.lineno, stat.size / 1024))
        line = linecache.getline(frame.filename, frame.lineno).strip()
        if line:
            print('    %s' % line)

    other = top_stats[limit:]
    if other:
        size = sum(stat.size for stat in other)
        print("%s other: %.1f KiB" % (len(other), size / 1024))
    total = sum(stat.size for stat in top_stats)
    print("Total allocated size: %.1f KiB" % (total / 1024))


# tracemalloc.start()

# pred = MLP(100, [300, 300, 1])

# # start_time = time.time()

# example = Value(np.random.normal(size=(10, 100)))

# truth = Value(np.random.normal(size=(10, 100)))

# forward = pred(example)

# loss = ((forward - truth) ** 2).sum()

# loss.backward()

# snapshot = tracemalloc.take_snapshot()
# display_top(snapshot)

# print("--- %s seconds ---" % (time.time() - start_time))
