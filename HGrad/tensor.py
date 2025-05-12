import numpy as np

class Tensor:
    def __init__(self, data, _parents=(), _op='', label=''):
        # transform data into np.array, don't know if the type is correct
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

    def __add__(self, other):

        # if bias vector is one dimensional, make it a row vector
        if len(self.data.shape) > len(other.shape):
            other = np.expand_dims(other.data, axis=0)
        if len(self.data.shape) < len(other.shape):
            self.data = np.expand_dims(self.data, axis=0)

        other = other if isinstance(other, Tensor) else Tensor(other)

        out = Tensor(self.data + other.data, (self, other), '+')
        
        def _backward():
            # if the value summed was broadcasted, sum the upstream gradient over the rows
            # allows two possibilities: dimensions match (m, n) and (m, n)
            # first dimension is broadcastable (1, n) and (m, n)
            if out.grad.shape[0]==self.data.shape[0]: 
                self.grad += out.grad
            else:
                self.grad += out.grad.sum(axis=0)

            if out.grad.shape[0]==other.data.shape[0]: 
                other.grad += out.grad
            else:
                other.grad += out.grad.sum(axis=0)

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
        
        out = Tensor(self.data * other.data, (self, other), '*')

        def _backward():
            # if the value multiplied was broadcasted, sum the upstream gradient over the rows
            # allows two possibilities: dimensions match (m, n) and (m, n)
            # first dimension is broadcastable (1, n) and (m, n)
            if len(self.shape) == 0:
                self.grad += (out.grad * other.data).sum()
            elif out.grad.shape[0]==self.data.shape[0]: 
                self.grad += out.grad * other.data
            else:
                self.grad += (out.grad * other.data).sum(axis=0)

            if len(other.shape) == 0:
                other.grad += (out.grad * self.data).sum()
            elif out.grad.shape[0]==other.data.shape[0]: 
                other.grad += out.grad * self.data
            else:
                other.grad += (out.grad * self.data).sum(axis=0)

        out._backward = _backward
        return out

    def __rmul__(self, other):
        return self * other

    def __matmul__(self, other):

        other = other if isinstance(other, Tensor) else Tensor(other)

        out = Tensor(self.data @ other.data, (self, other), '@')

        def _backward():
            self.grad += out.grad @ other.data.T
            other.grad += self.data.T @ out.grad

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

    def unsqueeze(self, axis):

        out = Tensor(np.expand_dims(self.data, axis=axis), (self,))

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

        # working only for inputs with 2 dimensions, if 3 dimensions needs modifications:
        #if x.ndim == 2:
        #    dim = 0 # reduce over the batch dimension
        #elif x.ndim == 3:
        #    dim = (0,1) # reduce over the firts two dimensions
        #    # example: input is (32, 4, 20) => we want the batch norm
        #    # of only the 20 examples in the last dimension
        #xmean = x.mean(dim, keepdim=True)
        #xvar = x.var(dim, keepdim=True)

        m = self.data.shape[0]
        mu_b = np.mean(self.data, 0, keepdims=True)
        # ddof=1: use Bessel's correction
        var_b = np.var(self.data, 0, ddof=1, keepdims=True)
        numerator = self.data - mu_b
        denominator_inv = np.sqrt(var_b + epsilon) ** -1
        out = Tensor(numerator * denominator_inv, (self,), 'batch_norm')

        def _backward():
            self.grad += (denominator_inv / m) * (m * out.grad - out.grad.sum(0) - (m / (m-1)) * out.data * (out.grad * out.data).sum(0))

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


