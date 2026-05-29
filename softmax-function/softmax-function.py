import numpy as np

def softmax(x):
    x = np.asarray(x)

    if x.ndim == 1:
        exp_x = np.exp(x - np.max(x))
        return exp_x / np.sum(exp_x)

    elif x.ndim == 2:
        exp_x = np.exp(x - np.max(x, axis=1, keepdims=True))
        return exp_x / np.sum(exp_x, axis=1, keepdims=True)

    raise ValueError("Input must be a 1D or 2D array")