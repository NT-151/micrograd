## Scalar to Numpy Micrograd

The `micrograd` library was built for pedagogical purposes, and when applied to non-trivial problems such as MNIST or FashionMNIST it becomes extremely inefficient. One of the main reasons is the use of scalar Python objects as the fundamental data structure. 

Each scalar operation creates a new object, and the computational graph quickly becomes dominated by Python overhead rather than numerical computation.

To better understand how real deep-learning libraries work, I reimplemented micrograd using numpy ndarrays as the core data representation.



## The Core Change

The most important modification was changing the `data` attribute of a `Value` from a scalar (`float` / `int`) to an `ndarray`.

In the scalar version of micrograd, every arithmetic operation produces a new `Value` object, which leads to:

-   A large number of Python objects
    
-   Many small scalar operations
    
-   Significant memory and runtime overhead
    

Wrapping entire tensors inside a single `Value` object reduces both the number of operations and the number of objects created. The surviving operations are also more efficient due to characteristics of numpy[1]

----------

## Scalar vs Tensor Computation

In scalar micrograd, each node represents a function

$$f: \mathbb{R} \to \mathbb{R}$$

and each scalar stores it's own gradient object

Backpropagation follows the chain rule:

$$
\frac{d\mathcal{L}}{dW}=
\frac{d\mathcal{L}}{d\hat{y}}
\frac{d\mathcal{\hat{y}}}{d W}
$$

In code, this typically looks like:

`self.grad += other.data * out.grad` 

For addition.

When the data stored in each node becomes an ndarray, a node represents

$$
f: \mathbb{R}^n \to \mathbb{R}^m
$$
 
Which implies a Jacobian Matrix for the gradient.

For a vector valued function:
$$
y=f(x),
\;\;\;
f:\mathbb{R}^n 
\to \mathbb{R}^m
$$

The Jacobian can be defined as:

$$
J_f(x) = \begin{pmatrix}
\frac{\partial f_1}{\partial x_1} & \cdots & \frac{\partial f_1}{\partial x_n} \\
\vdots & \ddots & \vdots \\
\frac{\partial f_m}{\partial x_1} & \cdots & \frac{\partial f_m}{\partial x_n}
\end{pmatrix}
\in \mathbb{R}^{m\times n}
$$

This represents a collection of all the partial derivatives of $f(x)$ with respect to every component in input $x$.

However, explicitly constructing Jacobian matrices is infeasible in practice[2]. Modern autodiff systems rely on **vector-Jacobian products (VJPs)**.

Given a loss function $\mathcal{L}(y)$ the gradient with respect to the input $x$ is computed as:

$$
\frac{\partial \mathcal{L}}{\partial x} = J_f(x)^T \frac{\partial \mathcal{L}}{\partial y}
$$

Here, $\frac{\partial \mathcal{L}}{\partial y}$ is the upstream gradient, and each node only needs to implement how it transforms this vector locally. This logic already existed within micrograd, but had to be made shape-aware when working with ndarrays

## Broadcasting and Unbroadcasting

Switching to arrays also introduced **broadcasting**.

Broadcasting is required so that weights and biases can interact naturally with batched inputs during the forward pass. However, during backpropagation, gradients that were broadcast must be **reduced back to the original parameter shapes**.

To handle this, I implemented unbroadcasting: summing gradients along broadcasted dimensions so that parameter gradients match the original tensor shapes[3].



## Matrix Multiplication Gradients

Supporting matrix multiplication was essential for multilayer perceptrons. The gradients follow standard reverse-mode rules:

`dA = G @ B.T` 
`dB = A.T @ G `

where $G$ is the upstream gradient. 



## Performance and Memory

The performance difference between the scalar and NumPy versions was dramatic.

For a 4-layer MLP with 300 neurons per hidden layer, a very contrived example, yielded:

-   Scalar micrograd: **~3.28 seconds** per forward + backward pass
    
-   NumPy micrograd: **~0.0018 seconds**
    

Increasing the input dimension from 5 to 100 caused scalar micrograd to take 161 seconds, while the NumPy version finished in 0.00355 seconds.

Memory usage showed a similar pattern. For an MLP with architecture `(100, [300, 300, 1]`, memory usage was 2490.8 kib using the numpy, yet 861346.0 KiB with the original scalar based micrograd.

[1]https://numpy.org/doc/stable/user/whatisnumpy.html
[2]https://0xgeorgeassaad.github.io/blog/2025/backprop/#the-vjp-trick-never-materialize-jacobians
[3]http://coldattic.info/post/116/#:~:text=In%20order%20to%20backpropagate%20the,G