import numpy as np
from utils import vectorize
from instance import Backlog, LostSales
from utils import g


class PoissonDemand:
    def __init__(self):
        pass

    @vectorize
    def price(self, arrival_rate):
        return 0

    @vectorize
    def arrival_rate(self, price):
        return 0

    @staticmethod
    def arrival_rate_valid(rate):
        return rate >= 0

    def next_arrival(self, price):
        arrival_rate = self.arrival_rate(price)
        if self.arrival_rate_valid(arrival_rate):
            return np.random.exponential(1 / arrival_rate)
        else:
            return np.inf

    @property
    def max_price(self):
        return np.inf

    @property
    def min_price(self):
        return 0

    @property
    def min_rate(self):
        return self.arrival_rate(self.max_price)

    @property
    def max_rate(self):
        return self.arrival_rate(self.min_price)

    @property
    def max_profit(self):
        return 0

    def _dp_opt_rate(self, delta):
        """
        Solution to the convex problem: lambda^* = argmax_lambda lambda * (p(lambda) - delta)
        """
        FAKE_MAX = 1e6
        THRESHOLD = 1e-3

        def f(lamb):
            return lamb * (self.price(lamb) - delta)

        min_rate = self.min_rate
        max_rate = self.max_rate
        if max_rate == np.inf:
            max_rate = FAKE_MAX
        f_min = f(min_rate)
        f_max = f(max_rate)
        while max_rate - min_rate > THRESHOLD:
            f_1 = f(2 * min_rate / 3 + max_rate / 3)
            if f_min == f_1 == f_max:
                return min_rate
            f_2 = f(min_rate / 3 + 2 * max_rate / 3)
            if f_min < f_1 > f_2:
                max_rate = min_rate / 3 + 2 * max_rate / 3
                f_max = f_2
            elif f_1 < f_2 > f_max:
                min_rate = 2 * min_rate / 3 + max_rate / 3
                f_min = f_1
            elif f_1 == f_2:
                max_rate = min_rate / 3 + 2 * max_rate / 3
                f_max = f_2
                min_rate = 2 * min_rate / 3 + max_rate / 3
                f_min = f_1
            else:
                raise ValueError('Something went wrong, are we sure the optimization is concave?')
        return (max_rate + min_rate) / 2

    def dp_price_solve(self, curr_state, next_state):
        opt_price = self.price(self._dp_opt_rate(next_state - curr_state))
        if self.min_price <= opt_price <= self.max_price:
            return opt_price
        else:
            if next_state < curr_state:
                return self.min_price
            else:
                return self.max_price

    # def opt_static_price(self, instance, S):
    #     return self.dp_price_solve(0, K * self.arrival_rate(self.min_price) / S)

    @staticmethod
    def state_probabilities(arrival_rates):
        arrival_rates = np.array(arrival_rates)
        return (1 / arrival_rates) / np.sum(1 / arrival_rates)

    @staticmethod
    def cycle_rate(arrival_rates):
        arrival_rates = np.array(arrival_rates)
        return 1 / np.sum(1 / arrival_rates)

    @vectorize
    def opt_rate_given_profit(self, profit, holding):
        pass

    def h(self, S, profit):
        pass

    def pi(self, s, profit):
        pass

    def K(self, s, S, profit):
        if s < 0:
            dummy_instance = Backlog(pi=self.pi(s, profit), h=self.h(S, profit), K=0)
        else:
            dummy_instance = LostSales(h=self.h(S, profit), K=0)
        return sum([g(dummy_instance, self, i + 1, profit) for i in range(s, S)])

    def optimal_instance_zero_leadtime(self, s, S, profit):
        if s < 0:
            return Backlog(pi=self.pi(s, profit), h=self.h(S, profit), K=self.K(s, S, profit))
        else:
            return LostSales(h=self.h(S, profit), K=self.K(s, S, profit))


class LinearDemand(PoissonDemand):
    def __init__(self, a, b):
        super().__init__()
        self.a = a
        self.b = b

    @vectorize
    def arrival_rate(self, price):
        return self.a * (1 - price * self.b)

    @vectorize
    def price(self, arrival_rate):
        return (1 - arrival_rate / self.a) / self.b

    @property
    def max_price(self):
        return 1 / self.b

    @property
    def max_profit(self):
        return self.a / self.b / 4

    def _dp_opt_rate(self, delta):
        return self.a / 2 * (1 - self.b * delta)

    def opt_price(self, profit, holding):
        return (1 - np.sqrt(self.b * (holding + profit) / self.a)) / self.b

    @vectorize
    def opt_rate_given_profit(self, profit, holding):
        return np.sqrt(self.a * self.b * (holding + profit))

    def optimal_s(self, instance, profit):
        return int(np.floor((self.a / self.b / 4 - profit) / instance.holding_cost))

    def optimal_r(self, instance, profit):
        return int(np.ceil((self.a / self.b / 4 - profit) / instance.backlog_cost + 1))

    def h(self, S, profit):
        super(LinearDemand, self).h(S, profit)
        if S == 0:
            return np.inf
        return self.a / 4 / S / self.b - profit / S

    def pi(self, s, profit):
        super(LinearDemand, self).pi(s, profit)
        assert s <= 0
        if s >= -1:
            return np.inf
        return self.a / 4 / (-s - 1) / self.b - profit / (-s - 1)


class ExponentialDemand(PoissonDemand):
    def __init__(self, a, b):
        super().__init__()
        self.a = a
        self.b = b

    @vectorize
    def arrival_rate(self, price):
        return self.a * np.exp(- price * self.b)

    @vectorize
    def price(self, arrival_rate):
        if arrival_rate <= 1e-10:
            return np.inf
        return 1 / self.b * np.log(self.a / arrival_rate)

    @property
    def max_profit(self):
        return self.a / self.b / np.exp(1)

    def _dp_opt_rate(self, delta):
        return self.a * np.exp(-1 - self.b * delta)

    def opt_price(self, profit, holding):
        return 1 / self.b * np.log(self.a / (self.b * (holding + profit)))

    @vectorize
    def opt_rate_given_profit(self, profit, holding):
        return self.b * (holding + profit)

    def h(self, S, profit):
        super(ExponentialDemand, self).h(S, profit)
        if S == 0:
            return np.inf
        return self.a / np.exp(1) / S / self.b - profit / S

    def pi(self, s, profit):
        super(ExponentialDemand, self).pi(s, profit)
        if s == 0:
            return np.inf
        return self.a / np.exp(1) / (-s - 1) / self.b - profit / (-s - 1)