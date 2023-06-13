import numpy as np
from typing import Tuple, Union
import helper_functions as zero


class Result:
    def __init__(self, revenue, ordering, holding, backlogging=0):
        self._rev = revenue
        self._order = ordering
        self._hold = holding
        self._back = backlogging

    @property
    def holding(self):
        return self._hold

    @property
    def revenue(self):
        return self._rev

    @property
    def ordering(self):
        return self._order

    @property
    def costs(self):
        return self.holding + self.ordering

    @property
    def profit(self):
        return self._rev - self._order - self._hold - self._back

    def holding_cost_ratio(self, result2):
        return self.holding / result2.holding

    def __repr__(self):
        return "Profit: {}\nRevenue: {}\nOrdering Costs: {}\nHolding Costs: {}".format(self.profit, self._rev,
                                                                                       self._order, self._hold)


class Policy:
    def __init__(self, demand, instance, state_space):
        self.demand = demand
        self.instance = instance
        self.state_space = state_space

        self._prices = None
        self._rates = None
        self._policy_counter = 0

        self._static_result = None
        self._holding = None
        self._ordering = None
        self._revenue = None
        self._result = None

        self._static_result_counter = 0
        self._result_counter = 0

    @property
    def prices(self):
        if self._prices is not None:
            return self._prices
        else:
            raise AttributeError("prices not set")

    @property
    def rates(self):
        if self._rates is not None:
            return self._rates
        else:
            raise AttributeError("rates not set")

    @prices.setter
    def prices(self, prices):
        prices = np.array(prices)
        assert prices.shape == self.state_space.shape
        self._prices = np.copy(prices)
        self._rates = self.demand.arrival_rate(prices)
        self._policy_counter += 1

    @rates.setter
    def rates(self, rates):
        rates = np.array(rates)
        assert rates.shape == self.state_space.shape
        self._rates = np.copy(rates)
        self._prices = self.demand.price(rates)
        self._policy_counter += 1

    # TODO: holding, ordering, revenue should only be callable through the result, similar to static

    @property
    def holding_cost(self) -> Union[float, np.ndarray]:
        return self.result.holding

    @property
    def ordering_cost(self) -> float:
        return self.result.ordering

    @property
    def revenue(self) -> Union[float, np.ndarray]:
        return self.result.revenue

    @property
    def profit(self):
        assert self._prices is not None
        return self.revenue - self.holding_cost - self.ordering_cost

    @property
    def result(self) -> Result:
        assert self._prices is not None
        if self._policy_counter > self._result_counter:
            if self.state_space.size == 0:
                self._holding = 0
                self._ordering = 0
                self._revenue = 0
            else:
                revenue_rates = self._prices * self._rates
                revenue_rates[np.isnan(revenue_rates)] = 0
                self._revenue = np.sum(revenue_rates * self.state_space.state_dist(self.rates))
                self._ordering = self.instance.ordering(
                    self.state_space.order_size) * self.state_space.cycle_frequency(self.rates)
                self._holding = np.sum(
                    self.state_space.inventory_dist(self.rates) * self.instance.holding(self.state_space.inventory_states))
            self._result = Result(self._revenue, self._ordering, self._holding)
            self._result_counter = self._policy_counter
        return self._result

    @property
    def static_rate(self):
        assert self._prices is not None
        return self.state_space.static_rate(self.rates)

    @property
    def static_rates(self):
        assert self._prices is not None
        rate = self.static_rate
        static_rates = rate * np.ones(self.state_space.shape)
        return static_rates

    @property
    def static_result(self) -> Result:
        assert self._prices is not None
        if self._policy_counter > self._static_result_counter:
            curr_rates = np.copy(self._rates)
            self.rates = self.static_rates
            st_hold = self.holding_cost
            st_order = self.ordering_cost
            st_rev = self.revenue
            self._static_result = Result(st_rev, st_order, st_hold)
            self.rates = curr_rates
            self._static_result_counter = self._policy_counter
        return self._static_result

    @property
    def holding_cost_ratio(self):
        static_holding = self.static_result.holding
        dynamic_holding = self.result.holding
        return static_holding / dynamic_holding

    @property
    def ordering_cost_ratio(self):
        static_ordering = self.static_result.ordering
        dynamic_ordering = self.result.ordering
        return static_ordering / dynamic_ordering

    @property
    def revenue_ratio(self):
        static_revenue = self.static_result.revenue
        dynamic_revenue = self.result.revenue
        return static_revenue / dynamic_revenue

    def implement(self, time) -> Result:
        pass

    def relative_value_iteration(self, max_iters=300000, threshold=1e-5, debug=False, init=None) -> Tuple[
            float, np.ndarray, np.ndarray]:
        pass

    def solve_dp_and_set_policy(self, max_iters=300000, threshold=1e-5, debug=False, init=None) -> Tuple[
            float, np.ndarray, np.ndarray]:
        profit, prices, J = self.relative_value_iteration(max_iters, threshold, debug, init)
        self.prices = prices
        return profit, prices, J

    def set_optimal_policy(self):
        pass

    def optimal_rates(self, profit):
        pass


class ZeroLeadTimePolicy(Policy):
    def three_price_rates(self):
        assert self._prices is not None

    def implement(self, time) -> Result:
        inventory_level = self.state_space.S
        current_time = 0.
        iteration_num = 0
        holding = 0.
        ordering = 0.
        revenue = 0.
        while current_time <= time:
            if inventory_level <= self.state_space.s:
                ordering += self.instance.ordering(self.state_space.S - self.state_space.s)
                inventory_level = self.state_space.S
            price = self.prices[-self.state_space.s - 1 + inventory_level]
            next_arrival = self.demand.next_arrival(price)
            current_time = current_time + next_arrival
            revenue += price
            holding += self.instance.holding(inventory_level) * next_arrival
            inventory_level = inventory_level - 1
            iteration_num += 1
        return Result(revenue / current_time, ordering / current_time, holding / current_time)

    def relative_value_iteration(self, max_iters=300000, threshold=1e-5, debug=False, init=None) -> Tuple[
            float, np.ndarray, np.ndarray]:
        fixed_cost = self.instance.fixed_cost
        a = self.demand.arrival_rate(self.demand.min_price)
        if init is None:
            J = np.zeros(self.state_space.max_shape)
        else:
            J = init
        J_new = np.copy(J)
        best_prices = np.zeros(self.state_space.max_shape)
        best_prices_new = np.zeros(self.state_space.max_shape)

        # history = []

        for _ in range(max_iters):
            for ind, i in self.state_space.states():
                best_prices_new[ind] = self.demand.dp_price_solve(J[ind] / a,
                                                                  J[ind - 1] / a + fixed_cost * (
                                                                          i == self.state_space.s + 1))
                transition_prob = self.demand.arrival_rate(best_prices_new[ind]) / a
                J_new[ind] = self.instance.holding(i) + transition_prob * (
                        J[ind - 1] + a * fixed_cost * (i == self.state_space.s + 1) - a * best_prices_new[ind]) + (
                                     1 - transition_prob) * J[ind]
            # history.append(np.copy(best_prices))
            if np.sum(np.abs(J_new - J[0] - J)) <= J.size * threshold:
                # if np.sum(np.abs(best_prices_new - best_prices)) <= best_prices.size * threshold:
                break
            else:
                best_prices = np.copy(best_prices_new)
            J = np.copy(J_new) - J[0]

        return -J[0], best_prices_new, J

    def optimal_rates(self, profit):
        # assert demand_type in ["lin", "exp"]
        # a = self.demand.a
        # b = self.demand.b
        s = self.state_space.s
        S = self.state_space.S
        states = np.arange(s + 1, S + 1)
        rates = self.demand.opt_rate(profit, self.instance.holding(states))
        # rates = [self.demand.opt_rate(profit, self.instance.holding(i)) for i in states]
        # if demand_type == "lin":
        #     rates = np.sqrt(a * b * (self.instance.holding(states) + profit))
        # elif demand_type == "exp":
        #     rates = b * (self.instance.holding(states) + profit)
        # else:
        #     raise NotImplementedError("Only linear (lin) and exponential (exp) demand implemented")
        return rates

    def set_optimal_policy(self):
        profit = zero.optimal_profit(self.demand, self.instance, self.state_space.s, self.state_space.S)
        optimal_rates = self.optimal_rates(profit)
        self.rates = optimal_rates


class ErlangLeadTimePolicy(Policy):
    @property
    def static_rates(self):
        static_rates = super(ErlangLeadTimePolicy, self).static_rates
        static_rates[0, :] = 0  # TODO: ONLY WORKS FOR LOST SALES
        return static_rates

    def implement(self, time) -> Result:
        assert self.state_space.r is not None
        inventory_level = self.state_space.Q
        delivery_phase = 0
        current_time = 0.
        iteration_num = 0
        holding = 0.
        ordering = 0.
        revenue = 0.
        order_out = False
        state_time_spent = np.zeros(self.state_space.shape)
        orders = 0
        state_arrivals = np.zeros(self.state_space.shape)
        while current_time <= time:
            if inventory_level <= self.state_space.r and not order_out:
                order_out = True
            price = self.prices[inventory_level, delivery_phase]
            next_customer = self.demand.next_arrival(price)
            delivery_phase_time = np.random.exponential(
                1 / self.state_space.k / self.state_space.mu) if order_out else np.inf
            next_arrival = np.min([next_customer, delivery_phase_time])
            current_time += next_arrival
            state_time_spent[inventory_level, delivery_phase] += next_arrival
            iteration_num += 1
            holding += self.instance.holding(inventory_level) * next_arrival
            if next_arrival == next_customer:
                if inventory_level == 0:
                    revenue -= self.instance.lost_sales_penalty
                else:
                    revenue += price
                    inventory_level -= 1
                    state_arrivals[inventory_level, delivery_phase] += 1
            else:
                if delivery_phase == self.state_space.k - 1:
                    delivery_phase = 0
                    inventory_level += self.state_space.Q
                    orders += 1
                    ordering += self.instance.ordering(self.state_space.Q)
                    order_out = False
                    state_arrivals[inventory_level, delivery_phase] += 1
                elif 0 < delivery_phase < self.state_space.k - 1:
                    delivery_phase += 1
                    state_arrivals[inventory_level, delivery_phase] += 1
                else:
                    if inventory_level <= self.state_space.r:
                        delivery_phase += 1
                        state_arrivals[inventory_level, delivery_phase] += 1
        state_time_spent = state_time_spent / orders
        state_arrivals = state_arrivals / orders
        return Result(revenue / current_time, ordering / current_time, holding / current_time)

    def relative_value_iteration(self, max_iters=300000, threshold=1e-5, debug=False, init=None) -> Tuple[
            float, np.ndarray, np.ndarray]:
        fixed_cost = self.instance.fixed_cost
        a = self.demand.arrival_rate(self.demand.min_price)
        p_r = self.state_space.k * self.state_space.mu  # phase arrival rate

        if init is None:
            J = np.zeros(self.state_space.max_shape)
        else:
            J = init
        J_new = np.copy(J)
        best_prices = np.zeros(self.state_space.max_shape)
        best_prices_new = np.zeros(self.state_space.max_shape)
        for _ in range(max_iters):
            for i, j, x, y in self.state_space.states():
                if x >= self.state_space.max_r + 1 and y >= 1:
                    continue
                best_prices_new[i, j] = self.demand.dp_price_solve(J[i, j], J[i - 1, j])
                arrival_rate = self.demand.arrival_rate(best_prices_new[i, j])
                value_to_go_sale = (x > 0) * (- best_prices_new[i, j] + J[i - 1, j]) + (x == 0) * (
                        self.instance.lost_sales_penalty + J[i, j])
                value_to_go_phase = 0
                if self.state_space.k > 1:
                    if y == 0:
                        if x >= self.state_space.max_r + 1:
                            value_to_go_phase = J[i, 0]
                        else:
                            value_to_go_phase = min(fixed_cost + J[i, 1], J[i, 0])
                    elif 0 < y < self.state_space.k - 1:
                        value_to_go_phase = J[i, j + 1]
                    else:
                        value_to_go_phase = J[i + self.state_space.Q, 0]
                elif self.state_space.k == 1:
                    if x >= self.state_space.max_r + 1:
                        value_to_go_phase = J[i, 0]
                    else:
                        value_to_go_phase = min(fixed_cost + J[i + self.state_space.Q, 0], J[i, 0])
                J_new[i, j] = self.instance.holding(x) + arrival_rate * value_to_go_sale + (a - arrival_rate) * J[
                    i, j] + p_r * value_to_go_phase
            # if np.sum(np.abs(best_prices_new - best_prices)) <= best_prices.size * threshold:
            if np.sum(np.abs((J_new - J[self.state_space.Q, 0]) / (a + p_r) - J)) <= J.size * threshold:
                break
            else:
                if debug and _ % 100 == 0:
                    print(_, -J[self.state_space.Q, 0],
                          np.sum(np.abs((J_new - J[self.state_space.Q, 0]) / (a + p_r) - J)) / J.size,
                          np.sum(np.abs(best_prices_new - best_prices)) / best_prices.size)
                J = (np.copy(J_new) - J[self.state_space.Q, 0]) / (a + p_r)
            best_prices = np.copy(best_prices_new)
        best_prices_new[0, :] = self.demand.max_price
        return -J[self.state_space.Q, 0], best_prices_new, J

    def profit_to_go(self, i, profit):
        Q = self.state_space.Q
        a = self.demand.a
        b = self.demand.b
        h = self.instance.holding_cost
        K = self.instance.fixed_cost

        return Q / b - K - np.sum([zero.g(self.instance, self.demand, j, profit) for j in range(i + 1, i + Q + 1)])

    def find_r(self, profit):
        r_max = 1
        while self.profit_to_go(r_max, profit) > 0:
            r_max *= 2
        r_min = r_max // 2

        while r_min < r_max - 1:
            r = (r_max + r_min) // 2
            p_to_go = self.profit_to_go(r, profit)
            if p_to_go > 0:
                r_min = r
            elif p_to_go < 0:
                r_max = r
            else:
                return r
        return r_min

    def calc_r(self, J):
        K = self.instance.ordering(self.state_space.order_size)
        Q = self.state_space.Q
        r_max = self.state_space.max_r
        if self.state_space.k > 1:
            r = np.max(np.where(J[:, 1] + K < J[:, 0]))
        else:
            r = np.max(np.where(J[:r_max + 1] - J[Q:] - K > 0))
        return r

    def solve_dp_and_set_policy(self, max_iters=300000, threshold=1e-5, debug=False, init=None) -> Tuple[
            float, np.ndarray, np.ndarray]:
        profit, prices, J = self.relative_value_iteration(max_iters, threshold, debug, init)
        r = self.calc_r(J)
        self.state_space.set_r(r)
        self.prices = prices[:r + self.state_space.Q + 1, :]
        return profit, prices, J

    def set_optimal_policy(self):
        # TODO: This only works for linear demand. Idea outlined in Notebook 30_oct_22
        h = self.instance.holding_cost
        K = self.instance.fixed_cost
        Q = self.state_space.Q
        a = self.demand.a
        b = self.demand.b
        mu = self.state_space.mu
        # max_r = self.state_space.max_r

        rates = None
        r = None

        profit_max = self.demand.max_profit
        profit_min = 0
        while profit_max - profit_min > 0.00001:
            updated = False
            profit = (profit_max + profit_min) / 2

            # r = np.where(
            #     pre_order_curve(profit, max_r, h, self.demand) >= post_order_curve(profit, max_r, Q, h, self.demand,
            #                                                                        mu))[0][-1]
            # rates = np.r_[
            #     post_order_curve(profit, r, Q, h, self.demand, mu), pre_order_curve(profit, Q + r, h, self.demand)[
            #                                                         r + 1:]]

            # self.state_space.set_r(r)
            # self.rates = rates.reshape((-1, 1))
            # profit_rates = (self.rates * self.prices)[:, 0] - np.arange(Q + r + 1) * h - profit
            # K_apx = np.sum(self.state_space.state_time_spent(self.rates)[:, 0] * profit_rates)
            # if K_apx < K:
            #     profit_max = profit
            # else:
            #     profit_min = profit

            r = self.find_r(profit)

            J = np.zeros(r + Q + 1)
            rates = np.zeros(r + Q + 1)
            for i in range(r + 1, r + Q + 1):
                J[i] = J[i - 1] - zero.g(self.instance, self.demand, i, profit)
                rates[i] = min(a, self.demand.opt_rate(profit, self.instance.holding(i)))
            for i in range(r, 0, -1):
                inside_sqrt = a * b * (self.instance.holding(i) + profit) + mu * a * b * (K + J[i + Q] - J[i])
                if inside_sqrt < 0 and not updated:
                    profit_min = profit
                    updated = True
                    break
                rates[i] = min(a, np.sqrt(inside_sqrt))
                J[i - 1] = ((rates[i] + mu) * J[i] - mu * (J[i + Q] + K) - (i * h + profit)) / rates[i] + \
                    self.demand.price(rates[i])
            if updated:
                continue
            profit_check = mu * (J[0] - K - J[Q])
            if profit_check > profit:
                profit_min = profit
            elif profit_check < profit:
                profit_max = profit
            else:
                break
        self.state_space.set_r(r)
        self.rates = rates.reshape((-1, 1))