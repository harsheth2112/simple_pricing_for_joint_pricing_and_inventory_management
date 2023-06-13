import numpy as np


class InventorySpace:
    def __init__(self):
        self.max_shape = (1,)
        self.order_size = 1
        self.shape = self.max_shape
        self._rates = None
        self._time_spent = None

    def arrival_probs(self, rates):
        return np.ones(self.max_shape)

    def state_rates(self, rates):
        return rates

    def state_time_spent(self, rates):
        if not np.all(rates == self._rates):
            self._time_spent = self.arrival_probs(rates) / self.state_rates(rates)
            self._rates = np.copy(rates)
        return self._time_spent

    def state_dist(self, rates):
        q = self.state_time_spent(rates)
        return q / np.sum(q)

    def inventory_dist(self, rates):
        return self.state_dist(rates)

    def cycle_frequency(self, rates):
        return 1 / np.sum(self.state_time_spent(rates))

    def static_rate(self, rates):
        q = self.state_time_spent(rates)
        return np.sum(rates * q) / np.sum(q)


class ZeroLeadTime(InventorySpace):
    def __init__(self, s, S):
        super(ZeroLeadTime, self).__init__()
        self.s = s
        self.S = S
        self.max_shape = (int(self.S - self.s),)
        self.shape = self.max_shape
        self.order_size = self.S - self.s

    def states(self):
        for i, inv in enumerate(range(self.s, self.S)):
            yield i, inv + 1

    @property
    def inventory_states(self):
        return np.arange(self.s + 1, self.S + 1)

    @property
    def size(self):
        return self.S - self.s


class ErlangLeadTime(InventorySpace):
    def __init__(self, k, mu, Q, max_r=None):
        super(ErlangLeadTime, self).__init__()
        self.k = k
        self.mu = mu
        self.Q = Q
        self.r = None
        self.max_r = max_r if max_r is not None else self.Q
        self.max_shape = (int(self.max_r + self.Q + 1), int(self.k))
        self.shape = self.max_shape
        self.order_size = self.Q

    @property
    def size(self):
        return self.shape[0] * self.shape[1]

    def states(self):
        for i, inv in enumerate(range(self.max_shape[0])):
            for j, phase in enumerate(range(self.max_shape[1])):
                yield i, j, inv, phase

    def set_r(self, r):
        self.r = r
        self.shape = (self.r + self.Q + 1, self.k)

    def inventory_dist(self, rates):
        state_dist = super(ErlangLeadTime, self).inventory_dist(rates)
        return np.sum(state_dist, axis=1)

    @property
    def inventory_states(self):
        assert self.r is not None
        return np.arange(self.r + self.Q + 1)

    @property
    def phase_rates(self):
        assert self.r is not None
        phase_rates = self.k * self.mu * np.tile(np.r_[np.ones(self.r + 1), np.zeros(self.Q)], (self.k, 1)).T
        return phase_rates
    
    def state_rates(self, rates):
        rates[0, :] = 0
        return rates + self.phase_rates
    
    def static_rate(self, rates):
        q = self.state_time_spent(rates)
        q = q[1:, :]
        rates = rates[1:, :]
        return np.sum(rates * q) / np.sum(q)

    def arrival_probs(self, rates):
        assert self.r is not None
        _ = super(ErlangLeadTime, self).arrival_probs(rates)
        phase_probs = self.phase_rates / (rates + self.phase_rates)
        if self.r <= self.Q:
            probs = np.zeros(self.shape)
            probs[self.Q, 0] = 1
            for x in range(self.k):
                for i in reversed(range(self.Q + 1)):
                    if x < self.k - 1:
                        if i == 0:
                            probs[i, x + 1] += probs[i, x]
                        else:
                            probs[i, x + 1] += probs[i, x] * phase_probs[i, x]
                            probs[i - 1, x] += probs[i, x] * (1 - phase_probs[i, x])
                    else:
                        if i >= 1:
                            probs[i - 1, x] += probs[i, x] * (1 - phase_probs[i, x])
                            if i <= self.r:
                                probs[i + self.Q, 0] += probs[i, x] * phase_probs[i, x]
                        else:
                            continue

            for i in reversed(range(self.Q + 2, self.r + self.Q + 1)):
                probs[i - 1, 0] += probs[i, 0] * (1 - phase_probs[i, 0])
            return probs
        else:
            if self.k == 1:
                transition_mat = np.zeros((self.Q + self.r + 1, self.Q + self.r + 1))
                for i in range(self.r + 1):
                    transition_mat[i, i + self.Q] = phase_probs[i, 0]
                for i in range(1, self.Q + self.r + 1):
                    transition_mat[i, i - 1] = 1 - phase_probs[i, 0]
                transition_mat = np.c_[transition_mat - np.identity(self.Q + self.r + 1), np.ones((self.Q + self.r + 1, 1))]
                probs = np.linalg.inv(transition_mat @ transition_mat.T).T @ np.ones(self.shape) * (self.Q + self.k)
                return probs
            else:
                print("arrival probs unsolved for this case")

