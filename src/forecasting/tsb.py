import numpy as np


def tsb_forecast(demand, alpha=0.1, beta=0.1):
    demand = np.asarray(demand, dtype=float)
    if len(demand) == 0:
        return 0.0

    probability = float(demand[0] > 0)
    size = float(demand[0])

    for value in demand[1:]:
        occurrence = float(value > 0)
        probability = beta * occurrence + (1 - beta) * probability
        if value > 0:
            size = alpha * value + (1 - alpha) * size

    return probability * size
