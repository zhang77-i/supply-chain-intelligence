"""Cost-sensitive inventory decision model.

Newsvendor converts demand uncertainty into inventory decisions by balancing
overstock and stockout costs.
"""

from scipy.stats import norm


def calculate_quantile(underage_cost, overage_cost):
    critical_ratio = underage_cost / (underage_cost + overage_cost)
    return critical_ratio


def newsvendor_quantity(mean, std, underage_cost, overage_cost):
    ratio = calculate_quantile(underage_cost, overage_cost)
    z = norm.ppf(ratio)
    return max(0, mean + z * std)
