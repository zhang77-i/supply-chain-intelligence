def critical_ratio(underage_cost, overage_cost):
    if underage_cost <= 0 or overage_cost <= 0:
        raise ValueError("cost parameters must be positive")
    return underage_cost / (underage_cost + overage_cost)


def optimal_quantile(quantile_forecast, underage_cost, overage_cost):
    ratio = critical_ratio(underage_cost, overage_cost)
    index = min(len(quantile_forecast) - 1, int(ratio * len(quantile_forecast)))
    return quantile_forecast[index]
