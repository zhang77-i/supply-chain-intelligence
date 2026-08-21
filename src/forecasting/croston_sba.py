import numpy as np


def croston_sba_forecast(demand, alpha=0.1):
    demand = np.asarray(demand, dtype=float)
    non_zero = demand[demand > 0]

    if len(non_zero) == 0:
        return 0.0

    size = non_zero[0]
    interval = 1.0
    last = -1

    for idx, value in enumerate(demand):
        if value > 0:
            if last >= 0:
                interval = alpha * (idx - last) + (1 - alpha) * interval
            size = alpha * value + (1 - alpha) * size
            last = idx

    return (1 - alpha / 2) * size / interval
