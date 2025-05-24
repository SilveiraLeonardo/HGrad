from .random import normal
from .tensor import Tensor
import numpy as np

# to be added
#class Module:
#    def __init__(self):
#        self._parameters = {}   # name -> Tensor
#        self._modules    = {}   # name -> Module
#        self.training    = True
#
#    def __setattr__(self, name, value):
#        # if you do self.foo = Tensor, register it as a parameter
#        if isinstance(value, Tensor):
#            self._parameters[name] = value
#        # if you do self.bar = Module, register it as a sub‐module
#        elif isinstance(value, Module):
#            self._modules[name] = value
#        super().__setattr__(name, value) # do the default behavior 
#
#    def parameters(self):
#        p_list = []
#        # yield own parameters
#        for p in self._parameters.values():
#            p_list.append(p)
#        # then recurse into sub‐modules
#        for m in self._modules.values():
#            p_list += m.parameters()
#        return p_list
#
#    def train(self):
#        self.training = True
#        for m in self._modules.values():
#            m.train()
#
#    def eval(self):
#        self.training = False
#        for m in self._modules.values():
#            m.eval()
#
#    def num_parameters(self):
#        return sum(p.nelement() for p in self.parameters())

class Linear:
    def __init__(self, fan_in, fan_out, bias=True, gain=1.0, label_weight='', label_bias=''):

        self.weight = Tensor(np.random.normal(0.0, 1.0, (fan_in, fan_out)) * (gain / (fan_in ** 0.5)))
        self.weight.label = label_weight

        if bias:
            self.bias = Tensor(np.random.normal(0.0, 1.0, (1, fan_out)) * 0.1)
            self.bias.label = label_bias
        else:
            self.bias = None

    def __call__(self, x):
        self.out = x @ self.weight
        if self.bias is not None:
            self.out += self.bias
        return self.out

    def parameters(self):
        return [self.weight] + ([] if self.bias is None else [self.bias])

    def nelement(self):
        return int(sum(np.prod(p.shape) for p in self.parameters()))

class Embedding:
    def __init__(self, num_embedding, embedding_dim, label="Embeddings"):
        self.weight = normal((num_embedding,embedding_dim))
        self.weight.label = label

    def __call__(self, IX):
        # receive the indices and look up in the index table
        self.out = self.weight[IX]
        return self.out

    def parameters(self):
        return [self.weight]
    
    def nelement(self):
        return int(sum(np.prod(p.shape) for p in self.parameters()))

class Flatten:
    def __call__(self, x):
        self.out = x.view((x.shape[0], -1))

        return self.out

    def parameters(self):
        return []
    
    def nelement(self):
        return 0

class FlattenConsecutive:
    def __init__(self, n):
        self.n = n # number of elements is last dimension

    def __call__(self, x):
        B, T, C = x.shape

        x = x.view((B, T//self.n, C*self.n))

        if x.shape[1] == 1:
            # if T//self.n is 1, remove this dimension
            x = x.squeeze(1)

        self.out = x

        return self.out

    def parameters(self):
        return []
    
    def nelement(self):
        return 0

class PReLU:
    def __init__(self, init=0.25):
        self.a = Tensor(init, label='a')

    def __call__(self, x):
        self.out = x.prelu(self.a)

        return self.out

    def parameters(self):
        return [self.a]
    
    def nelement(self):
        return int(sum(np.prod(p.shape) for p in self.parameters()))

class Tanh:
    def __call__(self, x):
        self.out = x.tanh()

        return self.out

    def parameters(self):
        return []
    
    def nelement(self):
        return 0

class BatchNorm:
    def __init__(self, n_hidden, eps=1e-6, momentum=0.1):
        self.eps = eps
        self.momentum = momentum
        self.training = True

        self.gamma = Tensor(np.ones((1, n_hidden)), label='gamma')
        self.beta = Tensor(np.zeros((1, n_hidden)), label='beta')

        self.running_mean = np.ones((1, n_hidden))
        self.running_var = np.zeros((1, n_hidden))

    def __call__(self, x):

        if self.training:
            x_bn, x_mu, x_var = x.batch_norm(self.eps)

            self.running_mean = (1 - self.momentum) * self.running_mean + self.momentum * x_mu
            self.running_var = (1 - self.momentum) * self.running_var + self.momentum * x_var

        else:
            x_mu = self.running_mean
            x_var = self.running_var

            x_bn = Tensor((x.data - x_mu) / (np.sqrt(x_var + self.eps)))

        self.out = x_bn * self.gamma + self.beta

        return self.out

    def parameters(self):
        return [self.gamma, self.beta]
    
    def nelement(self):
        return int(sum(np.prod(p.shape) for p in self.parameters()))

class LayerNorm:
    def __init__(self, n_hidden, eps=1e-6):
        self.eps = eps

        self.gamma = Tensor(np.ones((1, n_hidden)), label='gamma')
        self.beta = Tensor(np.zeros((1, n_hidden)), label='beta')

    def __call__(self, x):

        x_ln = x.layer_norm(self.eps)
        self.out = x_ln * self.gamma + self.beta

        return self.out

    def parameters(self):
        return [self.gamma, self.beta]
    
    def nelement(self):
        return int(sum(np.prod(p.shape) for p in self.parameters()))

class Dropout:
    def __init__(self, p=0.5):
        self.p = p
        self.training = True

    def __call__(self, x):

        if self.training == False or self.p == 0:
            return x
        out = x.dropout(self.p)

        return out
