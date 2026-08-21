import numpy as np


def safety_stock(demand_std, lead_time, service_level_z):
    return service_level_z * demand_std * np.sqrt(lead_time)


def reorder_point(mean_demand, lead_time, safety_stock_value):
    return mean_demand * lead_time + safety_stock_value
