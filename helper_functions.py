import numpy as np
from instance import Backlog, LostSales
from policy import Result
from demand import LinearDemand, ExponentialDemand
from utils import g


# Basic metric computation functions
def total_holding(s, S, instance, rates):
    one = sum([instance.holding(i + 1) / rates[i - s] for i in range(s, S)])
    two = np.sum(1 / rates)
    return one / two


def total_order(instance, rates):
    return instance.ordering(0) / np.sum(1 / rates)


def total_revenue(demand, rates):
    return np.sum([demand.price(rate) for rate in rates]) / np.sum(1 / rates)


def total_profit(s, S, demand, instance, rates):
    return total_revenue(demand, rates) - total_holding(s, S, instance, rates) - total_order(instance, rates)


def total_result(s, S, demand, instance, rates):
    return Result(total_revenue(demand, rates), total_order(instance, rates), total_holding(s, S, instance, rates))


def holding_cost_ratio(s, S, instance, rates1, rates2):
    a = total_holding(s, S, instance, rates1)
    b = total_holding(s, S, instance, rates2)
    return b / a


# Functions for known profit rate
def optimal_pricing_policy_given_profit_and_inventory_policy(s, S, demand, instance, profit):
    """
    Optimal pricing policy when optimal rate given profit can be solved for demand function
    """
    return np.array([demand.opt_rate_given_profit(profit, instance.holding(i + 1)) for i in range(s, S)])


def optimal_inventory_policy_given_profit(demand, instance, profit):
    """
    Compute optimal s, S for a given instance and guess for profit.
    Using Binary search.
    """
    S = 1
    while g(instance, demand, S, profit) > 0:
        S *= 2
    S_max = S
    S_min = S // 2
    while S_max - S_min > 1:
        S = (S_max + S_min) // 2
        if g(instance, demand, S, profit) >= 0:
            S_min = S
        else:
            S_max = S
    # If backlog instance, calculate s* as well
    if isinstance(instance, Backlog):
        s = -1
        while g(instance, demand, s, profit) > 0:
            s *= 2
        s_min = s
        s_max = s // 2
        while s_max - s_min > 1:
            s = (s_max + s_min) // 2
            if g(instance, demand, s, profit) >= 0:
                s_min = s
            else:
                s_max = s
        return s_max, S_min
    else:
        return S_min


def optimal_inventory_policy_and_surplus_given_profit(demand, instance, profit):
    """
    Compute optimal s, S as well as total surplus (sum g).
    Using linear search.
    """
    S = 1
    total_surplus = 0
    surplus = g(instance, demand, S, profit)
    while surplus > 0:
        total_surplus += surplus
        S += 1
        surplus = g(instance, demand, S, profit)
    # If backlog instance, calculate s* as well
    if isinstance(instance, Backlog):
        s = -1
        surplus = g(instance, demand, s, profit)
        while surplus > 0:
            total_surplus += surplus
            s -= 1
            surplus = g(instance, demand, s, profit)
        return (s, S-1), total_surplus
    else:
        return S - 1, total_surplus


# Solving for optimal profit
def optimal_profit_and_inventory_policy(demand, instance, threshold=1e-10):
    """
    Compute optimal profit and optimal S.
    Using binary search on the guess for profit, linear search for S. Need linear search since the binary search
    for profit requires surplus computation.
    """
    K = instance.fixed_cost
    p_max = demand.max_profit
    p_min = 0
    while p_max - p_min > threshold:
        profit = (p_max + p_min) / 2
        inventory_policy, surplus = optimal_inventory_policy_and_surplus_given_profit(demand, instance, profit)
        if surplus < K:
            p_max = profit
        elif surplus > K:
            p_min = profit
        else:
            return profit, inventory_policy
    profit = (p_min + p_max) / 2
    return profit, optimal_inventory_policy_given_profit(demand, instance, profit)


def optimal_profit_given_inventory_policy(demand, instance, s, S, threshold=1e-10):
    K = instance.fixed_cost
    p_max = demand.max_profit
    p_min = 0
    while p_max - p_min > threshold:
        profit = (p_max + p_min) / 2
        surplus = sum([g(instance, demand, i, profit) for i in np.arange(s + 1, S + 1)])
        if surplus < K:
            p_max = profit
        elif surplus > K:
            p_min = profit
        else:
            return profit
    profit = (p_min + p_max) / 2
    return profit


def optimal_pricing_policy_given_inventory_policy(demand, instance, s, S):
    profit = optimal_profit_given_inventory_policy(demand, instance, s, S)
    return optimal_pricing_policy_given_profit_and_inventory_policy(s, S, demand, instance, profit)


def optimal_policy(demand, instance):
    profit, inventory_policy = optimal_profit_and_inventory_policy(demand, instance)
    s = None
    S = None
    if isinstance(instance, Backlog):
        s, S = inventory_policy
    elif isinstance(instance, LostSales):
        s = 0
        S = inventory_policy
    rates = optimal_pricing_policy_given_profit_and_inventory_policy(s, S, demand, instance, profit)
    return s, S, rates, profit


# Our Static policy computations
def static_policy_rate(rates):
    return 1 / np.mean(1 / rates)


def static_rates(rates):
    stat_rate = static_policy_rate(rates)
    return np.array([stat_rate for _ in rates])


def static_pricing_policy_for_instance(demand, instance):
    profit, inventory_policy = optimal_profit_and_inventory_policy(demand, instance)
    optimal_rates = None
    if isinstance(instance, LostSales):
        S = inventory_policy
        optimal_rates = optimal_pricing_policy_given_profit_and_inventory_policy(0, S, demand, instance, profit)
    elif isinstance(instance, Backlog):
        s, S = inventory_policy
        optimal_rates = optimal_pricing_policy_given_profit_and_inventory_policy(s, S, demand, instance, profit)
    return static_rates(optimal_rates)


# Optimal Static Policy for LostSales instance
def static_profit_given_rate_and_order_size(instance, demand, rate, S):
    assert isinstance(instance, LostSales)
    if S == 0:
        return 0
    K = instance.fixed_cost
    h = instance.holding_cost
    return rate * demand.price(rate) - K * rate / S - (S + 1) * h / 2


def optimal_static_profit_given_order_size(instance, demand, S):
    assert isinstance(instance, LostSales)
    return static_profit_given_rate_and_order_size(instance, demand,
                                                   optimal_static_rate_given_order_size(instance, demand, S), S)


def optimal_static_rate_given_order_size(instance, demand, S):
    assert isinstance(instance, LostSales)
    a = demand.a
    b = demand.b
    K = instance.fixed_cost
    if S == 0:
        return 0
    if isinstance(demand, LinearDemand):
        return max(0, a / 2 - a * b * K / 2 / S)
    elif isinstance(demand, ExponentialDemand):
        return a * np.exp(-1 - b * K / S)
    else:
        raise ValueError("Demand type not supported.")


def optimal_static_policy(instance, demand, threshold=1e-7):
    assert isinstance(instance, LostSales)
    K = instance.fixed_cost
    h = instance.holding_cost
    rate = demand.a * 0.99
    S = np.sqrt(2 * K * rate / h)
    p_0 = static_profit_given_rate_and_order_size(instance, demand, rate, S)
    p_1 = None

    i = 0
    while i == 0 or p_0 - p_1 > threshold:
        i += 1
        rate = optimal_static_rate_given_order_size(instance, demand, S)
        S = np.sqrt(2 * K * rate / h)
        p_1 = p_0
        p_0 = static_profit_given_rate_and_order_size(instance, demand, rate, S)
        # print(i, static_profit_comp(instance, demand, rate, S), rate, S)
    c = optimal_static_profit_given_order_size(instance, demand, np.ceil(S))
    f = optimal_static_profit_given_order_size(instance, demand, np.floor(S))
    if max(c, f) < 0:
        return 0, 0
    elif c > f:
        return optimal_static_rate_given_order_size(instance, demand, int(np.ceil(S))), int(np.ceil(S))
    else:
        return optimal_static_rate_given_order_size(instance, demand, int(np.floor(S))), int(np.floor(S))


# Other Simply Policies
def leave_zero_out_policy_given_rates(s, S, rates):
    rates_new = np.r_[rates[:-s - 1], rates[-s:]]
    rate_nonzero = static_policy_rate(rates_new)
    rate_zero = rates[-s - 1]
    return np.array([rate_nonzero for _ in range(-s - 1)] + [rate_zero] + [rate_nonzero for _ in range(S)])


def leave_zero_out_policy(demand, instance):
    s, S, rates, profit = optimal_policy(demand, instance)
    return s, S, leave_zero_out_policy_given_rates(s, S, rates)


def pos_neg_policy_given_rates(s, S, rates):
    rate_non_pos = static_policy_rate(rates[:-s])
    rate_pos = static_policy_rate(rates[-s:])
    return np.array([rate_non_pos for _ in range(-s)] + [rate_pos for _ in range(S)])


def pos_neg_policy(demand, instance):
    s, S, rates, profit = optimal_policy(demand, instance)
    return s, S, pos_neg_policy_given_rates(s, S, rates)


def three_price_policy_given_rates(s, S, rates):
    rate_neg = static_policy_rate(rates[:-s - 1])
    rate_zero = rates[-s - 1]
    rate_pos = static_policy_rate(rates[-s:])
    return np.array([rate_neg for _ in range(-s - 1)] + [rate_zero] + [rate_pos for _ in range(S)])


# Functions related to leadtime research
def post_order_curve(profit, max_r, Q, h, demand, mu):
    a = demand.a
    b = demand.b
    rates_plus = np.sqrt(a * b * ((Q + np.arange(1, max_r + 1)) * h + profit))
    rels = demand.price(rates_plus) - ((Q + np.arange(1, max_r + 1)) * h + profit) / rates_plus
    rates = np.zeros(max_r + 1)
    a_i = 0
    for i in range(1, max_r + 1):
        a_i = rates[i - 1] / (mu + rates[i - 1]) * (a_i - demand.price(rates[i - 1])) + (i - 1) * h / (
                mu + rates[i - 1]) + rels[i - 1]
        rates[i] = - mu + np.sqrt(mu ** 2 - a * b * mu * (a_i - 1 / b) + a * b * i * h)
    return rates


def pre_order_curve(profit, max_r, h, demand):
    a = demand.a
    b = demand.b
    rates = np.sqrt(a * b * (np.arange(1, max_r + 1) * h + profit))
    return np.r_[0, rates]
