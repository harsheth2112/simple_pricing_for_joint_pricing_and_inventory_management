from functools import wraps
import numpy as np


def vectorize(fn):
    vectorized = np.vectorize(fn, otypes=[float])

    @wraps(fn)
    def wrapper(*args):
        return vectorized(*args)
    return wrapper


def g(instance, demand, i, profit):
    """
    Surplus function
    """
    rate = demand.opt_rate_given_profit(profit, instance.holding(i))
    return demand.price(rate) - (instance.holding(i) + profit) / rate
