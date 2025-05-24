import numpy as np
from .tensor import Tensor 

def normal(shape):

    return Tensor(np.random.normal(0.0, 1.0, shape))

def multinomial(probs):

    batch_size = probs.shape[0]

    # for each example in the batch, draw a one-hot vector of length V
    samples = np.array([
        np.random.multinomial(1, probs.data[i])
        for i in range(batch_size)
    ])            # shape (B, V)

    # turn each one-hot into an index
    idx_next = samples.argmax(axis=1)   # shape (B,)

    return idx_next
