import random
from engine import Value
import math
import importlib
import numpy as np


class Module:

    def zero_grad(self):
        for p in self.parameters():
            p.grad = 0

    def zero_weights(self):
        for w in self.weights():
            w.grad = 0

    def parameters(self):
        return []

    def weights(self):
        return []

    def layers(self):
        return []

    def summary(self):
        return f"{len(self.layers())} layers, {len(self.parameters())} parameters"


class Neuron(Module):

    # I want to introduce weight sharing, which means I need to be able to
    # initialise a neuron with pre defined weights, but leave the bias?

    def __init__(self, nin, nonlin=True, **kwargs):
        self.tied = kwargs.get('tied', None)
        self.w = [Value(random.uniform(-1, 1))
                  for _ in range(nin)] if self.tied == None else self.tied[:nin]
        self.b = Value(random.uniform(-1, 1))
        self.nonlin = nonlin
        self.activate = kwargs.get('activate', None)

    def __call__(self, x):
        act = sum((wi*xi for wi, xi in zip(self.w, x)),
                  self.b) if len(self.w) > 1 else (self.w[0] * x) + self.b
        if self.activate and self.nonlin == False:
            return self.activate(act)
        else:
            return act.leaky_relu() if self.nonlin else act

    def parameters(self):
        return self.w + [self.b]

    def weights(self):
        return self.w

    def __repr__(self):
        return f"{'ReLU' if self.nonlin else '{self.activate}'}Neuron({len(self.w)})"


class Layer(Module):

    def __init__(self, nin, nout, **kwargs):
        self.neurons = [Neuron(nin, **kwargs) for _ in range(nout)]

    def __call__(self, x):
        out = [n(x) for n in self.neurons]
        return out[0] if len(out) == 1 else out

    def parameters(self):
        return [p for n in self.neurons for p in n.parameters()]

    def weights(self):
        return [p for n in self.neurons for p in n.weights()]

    def __repr__(self):
        return f"Layer of [{', '.join(str(n) for n in self.neurons)}]"


class MLP(Module):

    def __init__(self, nin, nouts, **kwargs):
        sz = [nin] + nouts
        self.layers = [Layer(sz[i], sz[i+1], nonlin=i != len(nouts)-1, **kwargs)
                       for i in range(len(nouts))]

    def __call__(self, x):
        for layer in self.layers:
            x = layer(x)
        return x

    def parameters(self):
        return [p for layer in self.layers for p in layer.parameters()]

    def weights(self):
        return [p for layer in self.layers for p in layer.weights()]

    def __repr__(self):
        return f"MLP of [{', '.join(str(layer) for layer in self.layers)}]"


# class Encoder(Module):
#     def __init__(self, )


# class Decoder(Module):
    # def __init__(self, )

class AutoEncoder(Module):
    # i want this to be used so that you give it input and then you will get the
    # exact output that you used, so it will undo all the activation functions
    # but how to do?

    # self.weights() is the whole list of weights for the whole neural network
    # but i only want to pass it the weights for that layer i believe. how do
    # i do that?

    def __init__(self, in_embeds, n_hidden_layers, compressed, act_func=None, tied=False):
        # determining the number of hidden layers, which should be reduced the further you go
        n_hidden_layers = [math.ceil(in_embeds / i)
                           for i in range(2, n_hidden_layers + 2)]
        # create the layers that are not the input layer which
        # equal to hidden_layers plus the final compressed latent representation
        # layer
        self.encoder = MLP(in_embeds, (n_hidden_layers + [compressed]))
        # for the decoder do it backwards, because you're mean to build it in reverse
        # the activation function of the final layer should be a sigmoid
        # so pass it in which will go up the chain to the neuron class
        # and call it on the final layer outputs
        self.act_func = act_func
        if tied == True:
            self.decoder = MLP(
                compressed, (list(reversed(n_hidden_layers)) + [in_embeds]), tied=np.array(self.weights()).T, activate=act_func)
        if tied == False:
            self.decoder = MLP(
                compressed, (list(reversed(n_hidden_layers)) + [in_embeds]), activate=act_func)

    def __call__(self, x):
        # ze architecture vurks like you give it unt input,
        # it passes it into ze encoder, zen it compresses
        # zen it passes it into ze decoder which has activation function
        # in ze final layer then it produces output, (i was being german for the last 5 minutes yesterday, yesterday being 26/11/25)

        compressed = self.encoder(x)
        out = self.decoder(compressed)
        return out

    def parameters(self):
        return self.encoder.parameters() + self.decoder.parameters()

    def weights(self):
        return self.encoder.weights()

    def layers(self):
        return self.encoder.layers + self.decoder.layers

    def pretty(self):
        if self.act_func != None:
            hey = str(self.act_func)
            return hey.split()[1][6:]
        else:
            return "no function"

    def __repr__(self):
        return f"encoder has {self.summary()}, decoder has {self.summary()} activated with {self.pretty()}"


hey = [1.0, 2.0, 3.0, 4.0]

yb = [1.0, 0.0, 0.0, 1.0]

auto = AutoEncoder(4, 1, 1, Value.sigmoid)


# print(auto.weights()[0])


# is it done? the autoencoder implementation?


# def trace(root):
#     nodes, edges = set(), set()

#     def build(v):
#         if v not in nodes:
#             nodes.add(v)
#             for child in v._prev:
#                 edges.add((child, v))
#                 build(child)
#     build(root)
#     return nodes, edges


# def draw_dot(root, format='svg', rankdir='LR'):
#     """
#     format: png | svg | ...
#     rankdir: TB (top to bottom graph) | LR (left to right)
#     """
#     assert rankdir in ['LR', 'TB']
#     nodes, edges = trace(root)
#     # , node_attr={'rankdir': 'TB'})
#     dot = Digraph(format=format, graph_attr={'rankdir': rankdir})

#     for n in nodes:
#         dot.node(name=str(id(n)), label="{ data %.4f | grad %.4f }" % (
#             n.data, n.grad), shape='record')
#         if n._op:
#             dot.node(name=str(id(n)) + n._op, label=n._op)
#             dot.edge(str(id(n)) + n._op, str(id(n)))

#     for n1, n2 in edges:
#         dot.edge(str(id(n1)), str(id(n2)) + n2._op)

#     return dot


# temp = auto(hey)


# mean = sum((yg - yp)**2 for yg, yp in zip(yb, temp))

# mean.backward()

# print(draw_dot(mean))
