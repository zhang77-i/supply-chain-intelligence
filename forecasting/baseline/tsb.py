import numpy as np


def tsb_forecast(demand, alpha=0.1, beta=0.1):
    demand = np.asarray(demand, dtype=float)
    if len(demand) == 0:
        return 0.0

    occurrence = float(demand[0] > 0)
    size = float(demand[0]) if demand[0] > 0 else 0.0

    for value in demand[1:]:
        occur = float(value > 0)
        occurrence = beta * occur + (1 - beta) * occurrence
        if value > 0:
            size = alpha * value + (1 - alpha) * size

    return occurrence * size
