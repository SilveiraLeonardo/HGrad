import numpy as np

class Tensor:
    def __init__(self, data, _parents=(), _op='', label=''):
        data = data if isinstance(data, type(np.array([]))) else np.array(data)
        
        self.data = np.ascontiguousarray(data)
        self.grad = np.zeros_like(data)
        self.shape = data.shape
        self._backward = lambda: None
        self._prev = set(_parents)
        self._op = _op
        self.label = label

    def __repr__(self):
        return f"{self.data}"

    def __getitem__(self, key):
        # subscriptable behavior
        out = Tensor(self.data[key], (self,), "_getitem")

        def _backward():
            np.add.at(self.grad, key, out.grad)

        out._backward = _backward

        return out

    def view(self, key):

        out = Tensor(np.ascontiguousarray(self.data.reshape(key)), (self,), 'view')

        def _backward():
            self.grad += out.grad.reshape(self.data.shape)

        out._backward = _backward

        return out
    
    def unsqueeze(self, axis):

        out = Tensor(np.expand_dims(self.data, axis=axis), (self,))
        
        def _backward():
            pass
        
        out._backward = _backward

        return out

    def squeeze(self, axis=None):
        out = Tensor(np.squeeze(self.data, axis=axis), (self,), 'squeeze')
        
        def _backward():
            self.grad += out.grad.reshape(self.data.shape)
        
        out._backward = _backward

        return out


    def __add__(self, other):

        other = other if isinstance(other, Tensor) else Tensor(other)        

        a, b = self.data, other.data

        # forward — numpy will broadcast a and b to the same shape
        out_data = a + b
        out = Tensor(out_data, (self, other), '+')

        def _backward():
            grad = out.grad
            # shapes
            a_shape = a.shape
            b_shape = b.shape
            out_shape = grad.shape

            # helper to find which axes to reduce over
            def _get_axes(orig_shape, result_shape):
                # pad orig_shape on the left with 1s to match len(result_shape)
                ndims = len(result_shape)
                padded = (1,) * (ndims - len(orig_shape)) + orig_shape
                # any axis where padded == 1 but result > 1 was broadcasted
                return tuple(i for i, (s_padded, s_out) in
                             enumerate(zip(padded, result_shape))
                             if s_padded == 1 and s_out > 1)

            # compute grad wrt b
            axes_b = _get_axes(b_shape, out_shape)
            grad_b = grad
            if axes_b:
                grad_b = grad.sum(axis=axes_b, keepdims=True)
            grad_b = grad_b.reshape(b_shape)
            other.grad += grad_b
            
            # compute grad wrt a
            axes_a = _get_axes(a_shape, out_shape)
            grad_a = grad
            if axes_a:
                grad_a = grad.sum(axis=axes_a, keepdims=True)
            # finally, reshape back to original a_shape in case we prepended dims
            grad_a = grad_a.reshape(a_shape)
            self.grad += grad_a

        out._backward = _backward
        return out

        out._backward = _backward

        return out

    def __radd__(self, other):
        return self + other

    def __sub__(self, other):
        other = other if isinstance(other, Tensor) else Tensor(other)

        out = Tensor(self.data - other.data, (self, other), '-')

        def _backward():
            pass

        out._backward = _backward
        return out

    def __rsub__(self, other):
        return -self + other

    def __neg__(self):
        return self * -1

    def __mul__(self, other):

        other = other if isinstance(other, Tensor) else Tensor(other)
        
        a, b = self.data, other.data

        out_data = a * b
        out = Tensor(out_data, (self, other), '*')

        def _backward():
            # we’ll make a tiny helper to accumulate grads
            def _acc_grad(var, var_data, other_data):
                # raw gradient w.r.t. var is upstream * the other operand’s data
                grad = out.grad * other_data

                # figure out which axes were broadcast in var
                out_shape = out_data.shape
                var_shape = var_data.shape
                # left-pad var_shape with 1’s to match len(out_shape)
                ndiff = len(out_shape) - len(var_shape)
                var_shape_ext = (1,) * ndiff + var_shape

                # any dimension where var_shape_ext==1 but out_shape>1 was broadcast
                axes = tuple(
                    i for i, (vs, os) in enumerate(zip(var_shape_ext, out_shape))
                    if vs == 1 and os > 1
                )
                if axes:
                    # sum out the broadcasted axes
                    grad = grad.sum(axis=axes)

                # reshape back to the original var shape
                grad = grad.reshape(var_shape)
                var.grad += grad

            # accumulate for self and other
            _acc_grad(self, self.data, other.data)
            _acc_grad(other, other.data, self.data)

        out._backward = _backward
        return out

    def __rmul__(self, other):
        return self * other

    def __matmul__(self, other):

        other = other if isinstance(other, Tensor) else Tensor(other)

        A = self.data
        B = other.data

        # forward
        C = A @ B
        out = Tensor(C, (self, other), '@')

        def _backward():
            G = out.grad  # ∂L/∂C, same shape as C

            # grad w.r.t. A: G @ B^T
            grad_A = G @ np.swapaxes(B, -1, -2)
            # grad w.r.t. B: A^T @ G
            grad_B = np.swapaxes(A, -1, -2) @ G
            # if A or B were broadcast in the forward pass, we need
            # to sum‐reduce the extra dims:

            if grad_A.shape != A.shape:
                # any leading dims in grad_A that A didn’t have
                extra_axes = tuple(range(grad_A.ndim - A.ndim))
                grad_A = grad_A.sum(axis=extra_axes)
            if grad_B.shape != B.shape:
                extra_axes = tuple(range(grad_B.ndim - B.ndim))
                grad_B = grad_B.sum(axis=extra_axes)

            self.grad += grad_A
            other.grad += grad_B

            #self.grad += out.grad @ other.data.T
            #other.grad += self.data.T @ out.grad

        out._backward = _backward

        return out
    
    def __rmatmul__(self, other):
        return self @ other 

    def __truediv__(self, other):
        other = other if isinstance(other, Tensor) else Tensor(other)

        out = Tensor(self.data / other.data, (self, other), '/')

        def _backward():
            pass

        out._backward = _backward
        return out

    def __rtruediv__(self, other):
        other = other if isinstance(other, Tensor) else Tensor(other)

        out = Tensor(other.data / self.data, (self, other), '/')

        def _backward():
            pass

        out._backward = _backward
        return out

    def __pow__(self, other):
        assert isinstance(other, (int, float))
        # only support int/float powers

        out = Tensor(self.data ** other, (self,), f'**{other}')

        def _backward():
            pass

        out._backward = _backward
        return out

    def log(self):

        out = Tensor(np.log(self.data), (self,), 'log')

        def _backward():
            pass

        out._backward = _backward
        return out

    def exp(self):

        out = Tensor(np.exp(self.data), (self,), 'exp')

        def _backward():
            pass

        out._backward = _backward
        return out

    def sum(self, axis=None, keepdims=False):

        out = Tensor(np.sum(self.data, axis=axis, keepdims=keepdims), (self,), 'sum')

        def _backward():
            pass

        out._backward = _backward
        return out

    def sigmoid(self):

        out = Tensor((1 / (1 + np.exp(-self.data))), (self,), 'sigmoid')
        
        def _backward():
            self.grad += (out.data * (1 - out.data)) * out.grad

        out._backward = _backward

        return out
    
    def relu(self):

        out = Tensor(np.maximum(0, self.data), (self,), 'relu')
        
        def _backward():
            self.grad += out.grad * (out.data > 0) * 1

        out._backward = _backward

        return out

    def prelu(self, a):

        out = Tensor(np.maximum(0, self.data) + np.minimum(a.data*self.data, 0), (self, a), 'prelu')

        def _backward():
            # to be checked
            self.grad += out.grad * (self.data > 0) * 1 + out.grad * (self.data <= 0) * a.data
            # supports only 'a' being a scalar
            a.grad += np.average(out.grad * (self.data <= 0) * self.data) # (b, hidden) 

        out._backward = _backward

        return out

    def tanh(self):

        out_data = (np.exp(self.data) - np.exp(-self.data)) / (np.exp(self.data) + np.exp(-self.data))
        out = Tensor(out_data, (self,), 'tanh')
        
        def _backward():
            self.grad += out.grad * (1 - out.data ** 2)

        out._backward = _backward

        return out

    def binary_cross_entropy_loss(self, y, epsilon=1e-7):
        n = self.shape[0]

        y = y.data if isinstance(y, Tensor) else y

        # the binary cross entropy is underfined and will return NaN
        # if its input is zero or one, so we offset it by a small epsilon to avoid this issue
        data = np.clip(self.data, epsilon, 1 - epsilon)

        if y.shape == (n,):
            y = np.expand_dims(y, axis=1)
        if self.shape == (n,):
            data = np.expand_dims(data, axis=1)

        out = y * (np.log(data)) + (1 - y) * np.log(1 - data)

        out = - np.sum(out) / n 

        out = Tensor(out, (self,), 'bce')

        def _backward():
            self.grad += -(1/n) * (y * data ** -1 - (1 - y) * (1 - data) ** -1)

        out._backward = _backward

        return out
   
    def batch_norm(self, epsilon=1e-6):

        # 1) decide which axes to reduce over:
        #    we will *not* reduce over the last axis,
        #    so: ex: if ndim=3: axes = (0, 1)
        axes = tuple(range(self.data.ndim - 1))

        # 2) forward pass
        mu_b = np.mean(self.data, axis=axes, keepdims=True)
        var_b = np.var(self.data, axis=axes, ddof=1, keepdims=True)
        inv_std = np.sqrt(var_b + epsilon) ** -1
        x_centered = self.data - mu_b
        out_data = x_centered * inv_std

        out = Tensor(out_data, (self,), 'batch_norm')

        # calculate the number of batches, multiplying all the batch dimensions
        m = np.prod([self.data.shape[a] for a in axes], dtype=float)

        def _backward():
            #dx = (1/N)·inv_std·(N·dL/dy – sum(dL/dy) – (N/(N-1)·y·sum(dL/dy·y))
            out_grad_sum = np.sum(out.grad, axis=axes, keepdims=True)
            out_grad_out_data_sum = np.sum(out.grad * out.data, axis=axes, keepdims=True)

            self.grad += (1.0/m) * inv_std * (m * out.grad - out_grad_sum - (m / (m-1)) * out.data * out_grad_out_data_sum)

        out._backward = _backward

        return out, mu_b, var_b

    def softmax(self):

        logit_maxes = self.data.max(1, keepdims=True) #(batch_size, 1)
        norm_logits = self.data - logit_maxes # (batch_size, classes) subtract max for numerical stability
        counts = np.exp(norm_logits) # (batch_size, classes)
        counts_sum = counts.sum(1, keepdims=True) # (batch_size, 1)
        counts_sum_inv = counts_sum ** -1 # (batch_size, 1)
        probs = counts * counts_sum_inv # (batch_size, classes)

        out = Tensor(probs, (self,), 'softmax')

        def _backward():
            pass

        out._backward = _backward

        return out

    def cross_entropy_loss(self, y):

        # softmax + cross entropy
        n = self.data.shape[0]
        logit_maxes = self.data.max(1, keepdims=True) #(batch_size, 1)
        norm_logits = self.data - logit_maxes # (batch_size, classes) subtract max for numerical stability
        counts = np.exp(norm_logits) # (batch_size, classes)
        counts_sum = counts.sum(1, keepdims=True) # (batch_size, 1)
        counts_sum_inv = counts_sum ** -1 # (batch_size, 1)
        probs = counts * counts_sum_inv # (batch_size, classes)
        logprobs = np.log(probs) # (batch_size, classes)
        loss = -logprobs[range(n), y].mean()

        out = Tensor(loss, (self,), 'cross_entropy_loss')

        def _backward():
            grad = probs
            grad[range(n), y] -= 1
            grad /= n
            self.grad += grad

        out._backward = _backward

        return out

    def zero_grad(self):
        
        topo = []
        visited = set()

        def build_topo(v):
            if v not in visited:
                visited.add(v)
                for child in v._prev:
                    build_topo(child)
                topo.append(v)

        self.grad = np.zeros_like(self.data)
        
        build_topo(self)

        for node in reversed(topo):
            node.grad = np.zeros_like(node.data)

    def backward(self):
        # calculate the gradients using backpropagation for all the nodes in the computation graph
        # up to that node you are calling from

        # find the topological ordering of the graph
        # the backprop will be performed in the opposite order
        topo = []
        visited = set()

        def build_topo(v):
            if v not in visited:
                visited.add(v)
                for child in v._prev:
                    build_topo(child)
                topo.append(v)

        # initilize the gradient for the node you are calling backprop from to 1.0
        self.grad = np.ones_like(self.data)

        build_topo(self)

        for node in reversed(topo):
            node._backward()


