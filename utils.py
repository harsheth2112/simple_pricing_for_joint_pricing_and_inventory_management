from functools import wraps
import numpy as np


def vectorize(fn):
    vectorized = np.vectorize(fn, otypes=[float])

    @wraps(fn)
    def wrapper(*args):
        return vectorized(*args)
    return wrapper
