from cainiao_inventory.schema import CONFIG_COLUMNS, ITEM_COLUMNS, STORE_COLUMNS


def test_competition_schema_widths() -> None:
    assert len(ITEM_COLUMNS) == 31
    assert len(STORE_COLUMNS) == 32
    assert len(CONFIG_COLUMNS) == 3
    assert ITEM_COLUMNS[-2] == "qty_alipay_njhs"
    assert STORE_COLUMNS[2] == "store_code"
