import numpy as np


def croston_forecast(demand, alpha=0.1):
    demand = np.asarray(demand, dtype=float)
    non_zero = np.where(demand > 0)[0]

    if len(non_zero) == 0:
        return 0.0

    z = demand[non_zero[0]]
    p = 1.0
    last = non_zero[0]

    for idx in non_zero[1:]:
        interval = idx - last
        z = alpha * demand[idx] + (1 - alpha) * z
        p = alpha * interval + (1 - alpha) * p
        last = idx

    return z / p
