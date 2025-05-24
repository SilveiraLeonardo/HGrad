import numpy as np

class SGD:
    def __init__(self, params, lr):
        self.params = params
        self.lr = lr

    def step(self):
        for p in self.params:
            p.data += -self.lr * p.grad

class SGDWithMomentum:
    def __init__(self, params, lr, momentum=0.9):
        self.params = params
        self.lr = lr
        self.mu = momentum

    def step(self):
        for p in self.params:
            p.velocity = self.mu * p.velocity - self.lr * p.grad
            p.data += p.velocity

class RMSProp:
    def __init__(self, params, lr, phi=0.9, epsilon=1e-8):
        self.params = params
        self.lr = lr
        self.phi = phi
        self.epsilon = epsilon

    def step(self):
        for p in self.params:
            p.s = self.phi * p.s + (1 - self.phi) * p.grad * p.grad
            p.data += -(self.lr / (np.sqrt(p.s + self.epsilon))) * p.grad

class Adam:
    def __init__(self, params, lr, beta1=0.9, beta2=0.999, epsilon=1e-8):
        self.params = params
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        self.t = 0

    def step(self):
        self.t += 1
        for p in self.params:

            p.velocity = self.beta1 * p.velocity + (1 - self.beta1) * p.grad
            p.s = self.beta2 * p.s + (1 - self.beta2) * p.grad * p.grad

            velocity_hat = p.velocity / (1 - self.beta1 ** self.t)
            s_hat = p.s / (1 - self.beta2 ** self.t)

            p.data += -(self.lr / (np.sqrt(s_hat) + self.epsilon)) * velocity_hat



