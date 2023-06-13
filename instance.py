from utils import vectorize
import numpy as np


class Instance:
    def __init__(self):
        self.fixed_cost = None

    def ordering(self, order):
        pass

    @vectorize
    def holding(self, state):
        pass


class LostSales(Instance):
    def __init__(self, K, h, pi=0):
        super(LostSales, self).__init__()
        self.fixed_cost = K
        self.holding_cost = h
        self.lost_sales_penalty = pi

    def ordering(self, order):
        return self.fixed_cost

    @vectorize
    def holding(self, state):
        return self.holding_cost * max(0, state)

    def __repr__(self):
        return "K = {:.2f}, h = {:.2f}, pi = {:.2f}".format(self.fixed_cost, self.holding_cost, self.lost_sales_penalty)


class Backlog(Instance):
    def __init__(self, K, h, pi=np.inf):
        super(Backlog, self).__init__()
        self.fixed_cost = K
        self.holding_cost = h
        self.backlog_cost = pi

    def ordering(self, order):
        return self.fixed_cost

    @vectorize
    def holding(self, state):
        if state > 0:
            return self.holding_cost * max(0, state)
        elif state < 0:
            return - self.backlog_cost * min(0, state)
        return 0

    def __repr__(self):
        return "K = {:.2f}, h = {:.2f}, b = {:.2f}".format(self.fixed_cost, self.holding_cost, self.backlog_cost)
