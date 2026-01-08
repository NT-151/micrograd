import numpy as np


class Value:
    def __init__(self, data, _children=(), _op=''):
        self.data = np.asarray(data, dtype=float) if not (
            isinstance(data, np.ndarray)) else data
        self.grad = np.zeros_like(self.data)
        self._backward = lambda: None
        self._prev = set(_children)
        self._op = _op  # the op that produced this node, for graphviz / debugging / etc

    @staticmethod
    def unbroadcast(grad, shape):
        grad = np.asarray(grad)
        # remove leading dims
        while grad.ndim > len(shape):
            grad = grad.sum(axis=0)
        # sum over axes where original had dim 1, i.e., the weights and biases
        for i, (g, s) in enumerate(zip(grad.shape, shape)):
            if s == 1 and g != 1:
                grad = grad.sum(axis=i, keepdims=True)
        return grad

    @classmethod
    def constant(cls, data):
        return cls(np.asarray(data), _op='constant')
    
    @property
    def T(self):
        out = Value(self.data.T, (self,), 'T')

        def _backward():
            self.grad += out.grad.T

        out._backward = _backward
        return out

    def __add__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        
        out = Value(self.data + other.data, (self, other), '+')
        print(self.data.shape, other.data.shape,out.data.shape,  "in addition")

        def _backward():
            self.grad += Value.unbroadcast(out.grad, self.data.shape)
            other.grad += Value.unbroadcast(out.grad, other.data.shape)
        out._backward = _backward
        return out

    def __mul__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        
        out = Value(self.data * other.data, (self, other), '*')
        print(self.data.shape, other.data.shape,out.data.shape, "in just standard mul")

        def _backward():
            self.grad += Value.unbroadcast(other.data *
                                           out.grad, self.data.shape)
            other.grad += Value.unbroadcast(self.data *
                                            out.grad, other.data.shape)
        out._backward = _backward
        return out

    def __pow__(self, other):
        assert isinstance(other, (int, float)
                          ), "only supporting int/float powers for now"
        out = Value(self.data**other, (self,), f'**{other}')

        def _backward():
            self.grad += (other * self.data**(other-1)) * out.grad
        out._backward = _backward

        return out

    def __getitem__(self, idx):
        out = Value(self.data[idx], (self,), 'slice')

        def _backward():
            grad = np.zeros_like(self.data)
            grad[idx] = out.grad
            self.grad += grad

        out._backward = _backward
        return out

    def relu(self):
        out = Value(np.maximum(0, self.data), (self,), 'relu')

        def _backward():
            self.grad += (self.data > 0) * out.grad
        out._backward = _backward
        return out

    def log(self):
        EPSILON = 1e-7
        clipped_data = np.maximum(EPSILON, self.data)

        out = Value(np.log(clipped_data), (self, ), 'log')

        def _backward():
            self.grad += (1 / clipped_data) * out.grad
        out._backward = _backward

        return out

    def exp(self):
        x = self.data
        out = Value(np.exp(x), (self, ), 'exp')

        def _backward():
            self.grad += out.data * out.grad
        out._backward = _backward

        return out

    def sigmoid(self):
        x = self.data
        t = 1 / (1 + (np.exp(-x)))

        out = Value(t, (self, ), 'sigmoid')

        def _backward():
            self.grad += (out.data * (1 - out.data)) * out.grad
        out._backward = _backward

        return out

    def sum(self, axis=None, keepdims=False):
        out = Value(np.sum(self.data, axis=axis,
                    keepdims=keepdims), (self,), 'sum')

        def _backward():
            g = out.grad
            if axis is not None and not keepdims:
                g = np.expand_dims(g, axis=axis)
            self.grad += np.ones_like(self.data) * g
        out._backward = _backward
        return out

    def __matmul__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data @ other.data, (self, other), '@')
        print(self.data.shape, other.data.shape,out.data.shape, "in mat mul")

        def _backward():
            a, b, g = self.data, other.data, out.grad
            
            # ensure all possible 1d arrays are 2d to ensure consistency
            # None inserts new axis of length 1 at specified position
            a2 = a[None, :] if a.ndim == 1 else a # (1,N)
            b2 = b[:, None] if b.ndim == 1 else b # (N,1)
            g2 = g[None, :] if g.ndim == 1 else g # (1,N)

            da2 = g2 @ b2.T
            db2 = a2.T @ g2

            self.grad += da2.reshape(a.shape)
            other.grad += db2.reshape(b.shape)

        out._backward = _backward
        return out

    def mean(self, axis=None, keepdims=False):
        denom = self.data.size if axis is None else self.data.shape[axis]
        return self.sum(axis=axis, keepdims=keepdims) * (1.0 / denom)

    def backward(self):

        # topological order all of the children in the graph
        topo = []
        visited = set()

        def build_topo(v):
            if v not in visited:
                visited.add(v)
                for child in v._prev:
                    build_topo(child)
                topo.append(v)
        build_topo(self)

        # go one variable at a time and apply the chain rule to get its gradient
        self.grad = np.ones_like(self.data)
        for v in reversed(topo):
            v._backward()

    def __ge__(self, other):
        return self.data >= other.data

    def __le__(self, other):
        return self.data <= other.data

    def __lt__(self, other):
        other_data = other.data if isinstance(other, Value) else other
        return self.data < other_data

    def __gt__(self, other):
        other_data = other.data if isinstance(other, Value) else other
        return self.data > other_data

    def __neg__(self):  # -self
        return self * -1

    def __radd__(self, other):  # other + self
        return self + other

    def __sub__(self, other):  # self - other
        return self + (-other)

    def __rsub__(self, other):  # other - self
        return other + (-self)

    def __rmul__(self, other):  # other * self
        return self * other

    def __truediv__(self, other):  # self / other
        return self * other**-1

    def __rtruediv__(self, other):  # other / self
        return other * self**-1

    def __repr__(self):
        return f"Value(data={self.data}, grad={self.grad})"

    



# h = np.array([
#     [2,3,4],
#     [1,2,3],
# ])

# x = np.array([
#     [1],
#     [2],
#     [3]
# ])
# print(h.shape)
# print(x.shape)

# t = h + x


# print(t)
