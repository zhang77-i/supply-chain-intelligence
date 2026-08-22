from __future__ import annotations

BASE_COLUMNS = [
    "date",
    "item_id",
    "cate_id",
    "cate_level_id",
    "brand_id",
    "supplier_id",
    "pv_ipv",
    "pv_uv",
    "cart_ipv",
    "cart_uv",
    "collect_uv",
    "num_gmv",
    "amt_gmv",
    "qty_gmv",
    "unum_gmv",
    "amt_alipay",
    "num_alipay",
    "qty_alipay",
    "unum_alipay",
    "ztc_pv_ipv",
    "tbk_pv_ipv",
    "ss_pv_ipv",
    "jhs_pv_ipv",
    "ztc_pv_uv",
    "tbk_pv_uv",
    "ss_pv_uv",
    "jhs_pv_uv",
    "num_alipay_njhs",
    "amt_alipay_njhs",
    "qty_alipay_njhs",
    "unum_alipay_njhs",
]

ITEM_COLUMNS = BASE_COLUMNS
STORE_COLUMNS = ["date", "item_id", "store_code", *BASE_COLUMNS[2:]]
CONFIG_COLUMNS = ["item_id", "store_code", "a_b"]

ID_COLUMNS = {
    "date": "BIGINT",
    "item_id": "BIGINT",
    "store_code": "VARCHAR",
    "cate_id": "BIGINT",
    "cate_level_id": "BIGINT",
    "brand_id": "BIGINT",
    "supplier_id": "BIGINT",
    "a_b": "VARCHAR",
}


def duckdb_column_map(columns: list[str]) -> str:
    typed = []
    for column in columns:
        column_type = ID_COLUMNS.get(column, "DOUBLE")
        typed.append(f"'{column}': '{column_type}'")
    return "{" + ", ".join(typed) + "}"
