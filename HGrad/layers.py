from .random import normal
from .tensor import Tensor

class Linear:
    def __init__(self, fan_in, fan_out, bias=True, gain=1.0, label_weight='', label_bias=''):

        self.weight = normal((fan_in, fan_out)) * (gain) / (fan_in) ** 0.5
        self.weight.label = label_weight 
        self.weight._prev = set()
        self.weight._op = ''

        if bias:
            self.bias = normal((1, fan_out)) * 0.1
            self.bias.label = label_bias
            self.bias._prev = set()
            self.bias._op = ''
        else:
            self.bias = None

    def __call__(self, x):
        self.out = x @ self.weight
        if self.bias is not None:
            self.out += self.bias
        return self.out

    def parameters(self):
        return [self.weight] + ([] if self.bias is None else [self.bias])

class PReLU:
    def __init__(self, init=0.25):
        self.a = Tensor(init, label='a')

    def __call__(self, x):
        self.out = x.prelu(self.a)

        return self.out

    def parameters(self):
        return [self.a]
