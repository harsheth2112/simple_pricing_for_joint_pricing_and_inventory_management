import numpy as np
from instance import Backlog, LostSales
from demand import LinearDemand, ExponentialDemand
from policy import Result


def g(instance, demand, i, profit):
    rate = demand.opt_rate(profit, instance.holding(i))
    return demand.price(rate) - (instance.holding(i) + profit) / rate


def total_holding(s, S, instance, rates):
    one = sum([instance.holding(i + 1) / rates[i - s] for i in range(s, S)])
    two = np.sum(1/rates)
    return one/two


def total_order(instance, rates):
    return instance.ordering(0) / np.sum(1/rates)


def total_revenue(demand, rates):
    return np.sum([demand.price(rate) for rate in rates]) / np.sum(1/rates)


def total_profit(s, S, demand, instance, rates):
    return total_revenue(demand, rates) - total_holding(s, S, instance, rates) - total_order(instance, rates)


def total_result(s, S, demand, instance, rates):
    return Result(total_revenue(demand, rates), total_order(instance, rates), total_holding(s, S, instance, rates))


def holding_cost_ratio(s, S, instance, rates1, rates2):
    a = total_holding(s, S, instance, rates1)
    b = total_holding(s, S, instance, rates2)
    return b / a


def optimal_dynamic_policy_known_profit(s, S, demand, instance, profit):
    return np.array([demand.opt_rate(profit, instance.holding(i + 1)) for i in range(s, S)])


def optimal_S(demand, instance, profit):
    """
    Compute optimal S (largest S for which g is still positive) for a given instance and guess for profit.
    Using Binary search.
    """
    i = 1
    while g(instance, demand, i, profit) > 0:
        i *= 2
    i_max = i
    i_min = i // 2
    while i_max - i_min > 1:
        i = (i_max + i_min) // 2
        if g(instance, demand, i, profit) >= 0:
            i_min = i
        else:
            i_max = i
    return i_min


def optimal_S_and_surplus(demand, instance, profit):
    """
    Compute optimal S as well as (sum g).
    Using linear search.
    """
    i = 1
    surplus = 0
    g_i = g(instance, demand, i, profit)
    while g_i > 0:
        surplus += g_i
        i += 1
        g_i = g(instance, demand, i, profit)
    return i-1, surplus


def optimal_profit_and_S(demand, instance, threshold=1e-10):
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
        S, surplus = optimal_S_and_surplus(demand, instance, profit)
        if surplus < K:
            p_max = profit
        elif surplus > K:
            p_min = profit
        else:
            return profit, S
    profit = (p_min + p_max) / 2
    return profit, optimal_S(demand, instance, profit)


def optimal_profit(demand, instance, s, S, threshold=1e-10):
    K = instance.fixed_cost
    p_max = demand.max_profit
    p_min = 0
    while p_max - p_min > threshold:
        profit = (p_max + p_min) / 2
        surplus = sum([g(instance, demand, i, profit) for i in np.arange(s+1, S+1)])
        if surplus < K:
            p_max = profit
        elif surplus > K:
            p_min = profit
        else:
            return profit
    profit = (p_min + p_max) / 2
    return profit


def static_policy_rate(rates):
    return 1 / np.mean(1 / rates)


def static_rates(rates):
    stat_rate = static_policy_rate(rates)
    return np.array([stat_rate for _ in rates])


def two_price_policy(s, S, rates):
    rates_new = np.r_[rates[:-s - 1], rates[-s:]]
    rate_nonzero = static_policy_rate(rates_new)
    rate_zero = rates[-s - 1]
    return np.array([rate_nonzero for _ in range(-s - 1)] + [rate_zero] + [rate_nonzero for _ in range(S)])


def two_price_policy_2(s, S, rates):
    rate_non_pos = static_policy_rate(rates[:-s])
    rate_pos = static_policy_rate(rates[-s:])
    return np.array([rate_non_pos for _ in range(-s)] + [rate_pos for _ in range(S)])


def three_price_policy(s, S, rates):
    rate_neg = static_policy_rate(rates[:-s - 1])
    rate_zero = rates[-s - 1]
    rate_pos = static_policy_rate(rates[-s:])
    return np.array([rate_neg for _ in range(-s - 1)] + [rate_zero] + [rate_pos for _ in range(S)])


# Optimal Static Policy
def static_profit_comp(instance, demand, l, s):
    if s == 0:
        return 0
    K = instance.fixed_cost
    h = instance.holding_cost
    return l * demand.price(l) -K*l/s - (s+1)*h/2


def optimal_static_profit_fixed_S(instance, demand, s):
    return static_profit_comp(instance, demand, optimal_static_rate_fixed_S(instance, demand, s), s)


def optimal_static_rate_fixed_S(instance, demand, s):
    a = demand.a
    b = demand.b
    K = instance.fixed_cost
    if s == 0:
        return 0
    if isinstance(demand, LinearDemand):
        return a / 2 - a * b * K / 2 / s
    elif isinstance(demand, ExponentialDemand):
        return a * np.exp(-1 - b * K / s)
    else:
        raise ValueError("Demand type not supported.")


def optimal_static_policy(instance, demand, threshold=1e-7):
    K = instance.fixed_cost
    h = instance.holding_cost
    l = 14.9
    S = np.sqrt(2 * K * l / h)
    p_0 = static_profit_comp(instance, demand, l, S)
    p_1 = None

    i = 0
    while i == 0 or p_0 - p_1 > threshold:
        i += 1
        l = optimal_static_rate_fixed_S(instance, demand, S)
        S = np.sqrt(2 * K * l / h)
        p_1 = p_0
        p_0 = static_profit_comp(instance, demand, l, S)
        # print(i, static_profit_comp(instance, demand, l, S), l, S)
    c = optimal_static_profit_fixed_S(instance, demand, np.ceil(S))
    f = optimal_static_profit_fixed_S(instance, demand, np.floor(S))
    if p_0 < 0:
        return 0, 0
    elif c > f:
        return optimal_static_rate_fixed_S(instance, demand, int(np.ceil(S))), int(np.ceil(S))
    else:
        return optimal_static_rate_fixed_S(instance, demand, int(np.floor(S))), int(np.floor(S))


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
