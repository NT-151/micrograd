
import numpy as np
import random
# from batch_iterator import BatchIterator
# History = {}


# class Value:
#     def __init__(self, data, _children=(), _op=''):
#         # print("this is data", type(data))
#         self.data = np.asarray(data, dtype=float) if not (
#             isinstance(data, np.ndarray)) else data
#         self.grad = np.zeros_like(self.data)
#         self._backward = lambda: None
#         self._prev = set(_children)
#         self._op = _op  # the op that produced this node, for graphviz / debugging / etc

#     @staticmethod
#     def unbroadcast(grad, shape):
#         grad = np.asarray(grad)
#         # remove leading dims
#         while grad.ndim > len(shape):
#             grad = grad.sum(axis=0)
#         # sum over axes where original had dim 1
#         for i, (g, s) in enumerate(zip(grad.shape, shape)):
#             if s == 1 and g != 1:
#                 grad = grad.sum(axis=i, keepdims=True)
#         return grad

#     @classmethod
#     def constant(cls, data):
#         return cls(np.asarray(data), _op='constant')

#     def __add__(self, other):
#         other = other if isinstance(other, Value) else Value(other)
#         out = Value(self.data + other.data, (self, other), '+')

#         def _backward():
#             self.grad += Value.unbroadcast(out.grad, self.data.shape)
#             other.grad = np.add(other.grad, Value.unbroadcast(
#                 out.grad, other.data.shape), out=other.grad, casting='unsafe')
#         out._backward = _backward
#         return out

#     def __mul__(self, other):
#         other = other if isinstance(other, Value) else Value(other)
#         out = Value(self.data * other.data, (self, other), '*')

#         def _backward():
#             self.grad += Value.unbroadcast(other.data *
#                                            out.grad, self.data.shape)
#             other.grad += Value.unbroadcast(self.data *
#                                             out.grad, other.data.shape)
#         out._backward = _backward
#         return out

#     def __pow__(self, other):
#         assert isinstance(other, (int, float)
#                           ), "only supporting int/float powers for now"
#         out = Value(self.data**other, (self,), f'**{other}')

#         def _backward():
#             self.grad += (other * self.data**(other-1)) * out.grad
#         out._backward = _backward

#         return out

#     def __getitem__(self, idx):
#         out = Value(self.data[idx], (self,), 'slice')

#         def _backward():
#             grad = np.zeros_like(self.data)
#             grad[idx] = out.grad
#             self.grad += grad

#         out._backward = _backward
#         return out

#     def relu(self):
#         out = Value(np.maximum(0, self.data), (self,), 'relu')

#         def _backward():
#             self.grad += (self.data > 0) * out.grad
#         out._backward = _backward
#         return out

#     def log(self):
#         EPSILON = 1e-7
#         clipped_data = np.maximum(EPSILON, self.data)

#         out = Value(np.log(clipped_data), (self, ), 'log')

#         def _backward():
#             self.grad += (1 / clipped_data) * out.grad
#         out._backward = _backward

#         return out

#     def exp(self):
#         x = self.data
#         out = Value(np.exp(x), (self, ), 'exp')

#         def _backward():
#             self.grad += out.data * out.grad
#         out._backward = _backward

#         return out

#     def sigmoid(self):
#         x = self.data
#         t = 1 / (1 + (np.exp(-x)))

#         out = Value(t, (self, ), 'sigmoid')

#         def _backward():
#             self.grad += (out.data * (1 - out.data)) * out.grad
#         out._backward = _backward

#         return out

#     def sum(self, axis=None, keepdims=False):
#         out = Value(np.sum(self.data, axis=axis,
#                     keepdims=keepdims), (self,), 'sum')

#         def _backward():
#             g = out.grad
#             if axis is not None and not keepdims:
#                 g = np.expand_dims(g, axis=axis)
#             self.grad += np.ones_like(self.data) * g
#         out._backward = _backward
#         return out

#     def __matmul__(self, other):
#         other = other if isinstance(other, Value) else Value(other)
#         out = Value(self.data @ other.data, (self, other), '@')

#         def _backward():
#             a, b, g = self.data, other.data, out.grad

#             a2 = a[None, :] if a.ndim == 1 else a
#             b2 = b[:, None] if b.ndim == 1 else b
#             g2 = g[None, :] if g.ndim == 1 else g

#             da2 = g2 @ b2.T
#             db2 = a2.T @ g2

#             self.grad += da2.reshape(a.shape)
#             other.grad += db2.reshape(b.shape)

#         out._backward = _backward
#         return out

#     def mean(self, axis=None, keepdims=False):
#         denom = self.data.size if axis is None else self.data.shape[axis]
#         return self.sum(axis=axis, keepdims=keepdims) * (1.0 / denom)

#     def backward(self):

#         # topological order all of the children in the graph
#         topo = []
#         visited = set()

#         def build_topo(v):
#             if v not in visited:
#                 visited.add(v)
#                 for child in v._prev:
#                     build_topo(child)
#                 topo.append(v)
#         build_topo(self)

#         # go one variable at a time and apply the chain rule to get its gradient
#         self.grad = np.ones_like(self.data)
#         for v in reversed(topo):
#             v._backward()

#     def __ge__(self, other):
#         return self.data >= other.data

#     def __le__(self, other):
#         return self.data <= other.data

#     def __lt__(self, other):
#         other_data = other.data if isinstance(other, Value) else other
#         return self.data < other_data

#     def __gt__(self, other):
#         other_data = other.data if isinstance(other, Value) else other
#         return self.data > other_data

#     def __neg__(self):  # -self
#         return self * -1

#     def __radd__(self, other):  # other + self
#         return self + other

#     def __sub__(self, other):  # self - other
#         return self + (-other)

#     def __rsub__(self, other):  # other - self
#         return other + (-self)

#     def __rmul__(self, other):  # other * self
#         return self * other

#     def __truediv__(self, other):  # self / other
#         return self * other**-1

#     def __rtruediv__(self, other):  # other / self
#         return other * self**-1

#     def __repr__(self):
#         return f"Value(data={self.data}, grad={self.grad})"


# class Module:

#     def zero_grad(self):
#         for p in self.parameters():
#             p.grad = np.zeros_like(p.data)

#     def parameters(self):
#         return []


# class Linear(Module):
#     def __init__(self, nin, nout):
#         k = np.sqrt(2.0 / nin)
#         self.W = Value(np.random.uniform(-k, k, (nin, nout)))
#         self.b = Value(np.full((nout,), 0.1))

#     def __call__(self, x):
#         # print("this is input", x.shape)
#         # print("this is weights", self.W.data.shape)
#         return (x @ self.W) + self.b

#     def parameters(self):
#         return [self.W, self.b]


# class MLP(Module):
#     def __init__(self, nin, nouts, act_func=None):
#         sz = [nin] + nouts
#         self.layers = [Linear(sz[i], sz[i+1]) for i in range(len(nouts))]
#         self.act_func = act_func

#     def __call__(self, x):
#         for i, layer in enumerate(self.layers):
#             x = layer(x)
#             if i != len(self.layers)-1:
#                 x = x.relu()
#             else:
#                 if self.act_func is not None:
#                     x = self.act_func(x)
#         return x

#     def parameters(self):
#         return [p for layer in self.layers for p in layer.parameters()]


# class AutoEncoder(Module):
#     def __init__(self, in_embeds=1, hidden_layers=[], latent_dim=1, act_func=None):
#         self.latent_dim = latent_dim
#         self.act_func = act_func
#         self.encoder = MLP(in_embeds, hidden_layers + [latent_dim])
#         self.decoder = MLP(latent_dim, list(
#             reversed(hidden_layers)) + [in_embeds], act_func=act_func)

#     def __call__(self, x):
#         compressed = self.encoder(x)
#         out = self.decoder(compressed)
#         return out

#     def encode(self, x):
#         encoded = self.encoder(x)
#         return encoded

#     def decode(self, x):
#         decoded = self.decoder(x)
#         return decoded

#     def parameters(self):
#         return self.encoder.parameters() + self.decoder.parameters()

#     def layers(self):
#         return self.encoder.layers + self.decoder.layers

#     def __repr__(self):
#         return f"encoder has {len(self.encoder.layers)}, decoder has {len(self.decoder.layers)}, latent dim is {self.latent_dim}"


# class VariationalAutoEncoder(Module):
#     def __init__(self, in_embeds, hidden_layers, latent_dim, act_func=None):
#         self.latent_dim = latent_dim
#         self.act_func = act_func

#         # encoder outputs 2*latent_dim
#         self.encoder = MLP(in_embeds, hidden_layers + [2 * latent_dim])
#         self.decoder = MLP(latent_dim, list(
#             reversed(hidden_layers)) + [in_embeds], act_func=act_func)

#     def encode(self, x: Value):
#         h = self.encoder(x)  # shape (B, 2d) or (2d,)

#         d = self.latent_dim
#         if h.data.ndim == 1:
#             mu = h[:d]
#             log_var = h[d:2*d]
#         else:
#             mu = h[:, :d]
#             log_var = h[:, d:2*d]
#         return mu, log_var

#     def reparameterize(self, mu: Value, log_var: Value):
#         # * inside randn() allows dynamic generation of numpy array
#         eps = Value.constant(np.random.randn(*mu.data.shape))
#         sigma = (log_var * 0.5).exp()
#         z = mu + sigma * eps
#         return z

#     def decode(self, z: Value):
#         return self.decoder(z)

#     def __call__(self, x: Value):
#         mu, log_var = self.encode(x)
#         z = self.reparameterize(mu, log_var)
#         recon = self.decode(z)
#         return recon, mu, log_var

#     def parameters(self):
#         return self.encoder.parameters() + self.decoder.parameters()


# class Optimizer:
#     """Base class for optimizers"""

#     def __init__(self, parameters):
#         self.parameters = parameters

#     def zero_grad(self):
#         for p in self.parameters:
#             p.grad = np.zeros_like(p.data)

#     def step(self):
#         """Take a step of gradient descent"""

#         raise NotImplementedError


# class SGD(Optimizer):
#     def __init__(self, parameters, learning_rate=0.01):
#         super().__init__(parameters)
#         self.learning_rate = learning_rate

#     def step(self):
#         for p in self.parameters:
#             p.data = p.data - (self.learning_rate * p.grad)


# def mean_squared_error(target, pred):
#     loss = ((pred - target) ** 2).sum()

#     return loss


# def vae_loss(recon, target, mu, log_var, beta=0.002):
#     # reconstruction loss (use mean to keep scales sane)
#     recon_loss = ((recon - target) ** 2).mean()

#     # kl per element
#     kl_elem = (Value(1.0) + log_var - (mu * mu) - log_var.exp())

#     # sum over latent dim; then mean over batch if batched
#     if kl_elem.data.ndim == 1:
#         kl = kl_elem.sum() * (-0.5)
#     else:
#         kl = kl_elem.sum(axis=1).mean() * (-0.5)

#     total_loss = recon_loss + beta * kl

#     # print("recon:", float(recon_loss.data), "kl:", float(kl.data), "total:", float(total_loss.data))
#     return total_loss


# class BatchIterator:
#     """Iterates on data by batches"""

#     def __init__(self, inputs, targets, batch_size=32, shuffle=True):
#         self.inputs = inputs.data
#         self.targets = targets.data
#         self.batch_size = batch_size
#         self.shuffle = shuffle

#     def __call__(self):
#         starts = list(range(0, len(self.inputs), self.batch_size))
#         if self.shuffle:
#             random.shuffle(starts)

#         for start in starts:
#             end = start + self.batch_size
#             batch_inputs = self.inputs[start:end]
#             batch_targets = self.targets[start:end]
#             yield (Value(batch_inputs), Value(batch_targets))


class Trainer:
    """Encapsulates the model training loop"""

    def __init__(self, model, optimizer, loss, name=None):
        self.model = model
        self.optimizer = optimizer
        self.loss = loss

    def fit(self, data_iterator, num_epochs=500, verbose=False):
        """Fits the model to the data"""

        history = {"loss": []}
        epoch_loss = 0
        epoch_y_true = []
        epoch_y_pred = []
        for epoch in range(num_epochs):
            epoch_loss = 0
            num_batches = 0
            epoch_y_true = []
            epoch_y_pred = []

            for batch in data_iterator():
                self.optimizer.zero_grad()
                outputs = self.model(batch[0])

                batch_loss = self.loss(batch[1], outputs)
                epoch_loss += batch_loss.data
                num_batches += 1

                epoch_y_pred.extend(outputs)
                epoch_y_true.extend(batch[1])

                batch_loss.backward()
                self.optimizer.step()

            avg_epoch_loss = epoch_loss / max(1, num_batches)
            history["loss"].append(avg_epoch_loss)
            if verbose:
                print(
                    f"Epoch [{epoch+1}/{num_epochs}], "
                    f"loss: {avg_epoch_loss:.6f}, "
                )

        return history

    def vae_fit(self, data_iterator, num_epochs=500, verbose=False):
        """Fits the model to the data"""

        history = {"loss": []}
        epoch_loss = 0
        epoch_y_true = []
        epoch_y_pred = []
        for epoch in range(num_epochs):
            epoch_loss = 0
            num_batches = 0
            epoch_y_true = []
            epoch_y_pred = []

            for batch in data_iterator():
                self.optimizer.zero_grad()
                recon, mu, log_var = self.model(batch[0])

                batch_loss = self.loss(recon, batch[1], mu, log_var)
                epoch_loss += batch_loss.data
                num_batches += 1

                epoch_y_pred.extend(recon)
                epoch_y_true.extend(batch[1])

                batch_loss.backward()
                self.optimizer.step()

            avg_epoch_loss = epoch_loss / max(1, num_batches)
            history["loss"].append(avg_epoch_loss)
            if verbose:
                print(
                    f"Epoch [{epoch+1}/{num_epochs}], "
                    f"loss: {avg_epoch_loss:.6f}, "
                )

        return history



# test = Value(np.random.normal(size=(100, 10)))


# auto = AutoEncoder(in_embeds=10, hidden_layers=[
#                    8, 6, 4], latent_dim=2, act_func=Value.sigmoid)
# optimizer = SGD(auto.parameters())
# data_iterator = BatchIterator(test, test)
# trainer = Trainer(auto, optimizer, loss=mean_squared_error)


# history = trainer.fit(data_iterator, num_epochs=50, verbose=True)
