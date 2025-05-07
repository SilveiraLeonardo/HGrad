import numpy as np
from .tensor import Tensor 

def normal(shape):

    return Tensor(np.random.normal(0.0, 1.0, shape))
