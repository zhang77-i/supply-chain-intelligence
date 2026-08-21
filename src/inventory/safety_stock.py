import math


def safety_stock(demand_std, lead_time, service_level_z=1.65):
    """Calculate safety stock under demand uncertainty.

    SS = z * sigma * sqrt(L)
    """
    return service_level_z * demand_std * math.sqrt(lead_time)


def reorder_point(mean_demand, lead_time, safety_stock_value):
    return mean_demand * lead_time + safety_stock_value
