from ortools.sat.python import cp_model


def build_multi_warehouse_model(items, warehouses, demand):
    model = cp_model.CpModel()

    allocation = {}
    for item in items:
        for warehouse in warehouses:
            allocation[item, warehouse] = model.NewIntVar(
                0, int(demand[item]),
                f"alloc_{item}_{warehouse}"
            )

    for item in items:
        model.Add(
            sum(allocation[item, w] for w in warehouses)
            >= int(demand[item])
        )

    return model, allocation
