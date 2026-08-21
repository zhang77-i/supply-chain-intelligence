CREATE OR REPLACE TABLE clean_item_feature AS
SELECT DISTINCT
    CAST(TRY_STRPTIME(CAST(raw.date AS VARCHAR), '%Y%m%d') AS DATE) AS date,
    raw.* EXCLUDE (date)
FROM raw_item_feature AS raw
WHERE TRY_STRPTIME(CAST(raw.date AS VARCHAR), '%Y%m%d') IS NOT NULL
  AND raw.item_id IS NOT NULL;

CREATE OR REPLACE TABLE clean_item_store_feature AS
SELECT DISTINCT
    CAST(TRY_STRPTIME(CAST(raw.date AS VARCHAR), '%Y%m%d') AS DATE) AS date,
    raw.item_id,
    CAST(raw.store_code AS VARCHAR) AS store_code,
    raw.* EXCLUDE (date, item_id, store_code)
FROM raw_item_store_feature AS raw
WHERE TRY_STRPTIME(CAST(raw.date AS VARCHAR), '%Y%m%d') IS NOT NULL
  AND raw.item_id IS NOT NULL
  AND raw.store_code IS NOT NULL;

CREATE OR REPLACE TABLE clean_config AS
WITH parsed AS (
    SELECT
        item_id,
        CAST(store_code AS VARCHAR) AS store_code,
        TRY_CAST(SPLIT_PART(a_b, '_', 1) AS DOUBLE) AS shortage_cost,
        TRY_CAST(SPLIT_PART(a_b, '_', 2) AS DOUBLE) AS overage_cost
    FROM raw_config
)
SELECT
    *,
    shortage_cost IS NOT NULL
        AND overage_cost IS NOT NULL
        AND shortage_cost >= 0
        AND overage_cost >= 0
        AND shortage_cost + overage_cost > 0 AS valid_cost_pair,
    CASE
        WHEN shortage_cost + overage_cost > 0
        THEN shortage_cost / (shortage_cost + overage_cost)
        ELSE NULL
    END AS critical_fractile
FROM parsed;

CREATE OR REPLACE TABLE dim_item AS
SELECT
    item_id,
    ARG_MAX(cate_id, date) AS cate_id,
    ARG_MAX(cate_level_id, date) AS cate_level_id,
    ARG_MAX(brand_id, date) AS brand_id,
    ARG_MAX(supplier_id, date) AS supplier_id
FROM clean_item_feature
GROUP BY item_id;

CREATE OR REPLACE TABLE fact_demand AS
SELECT
    date,
    item_id,
    'all'::VARCHAR AS store_code,
    'national'::VARCHAR AS scope,
    cate_id,
    cate_level_id,
    brand_id,
    supplier_id,
    qty_alipay_njhs AS demand,
    pv_ipv,
    pv_uv,
    cart_ipv,
    cart_uv,
    collect_uv,
    num_gmv,
    qty_gmv,
    unum_gmv,
    num_alipay,
    qty_alipay,
    unum_alipay,
    ztc_pv_uv,
    tbk_pv_uv,
    ss_pv_uv,
    jhs_pv_uv,
    num_alipay_njhs,
    qty_alipay_njhs,
    unum_alipay_njhs
FROM clean_item_feature

UNION ALL

SELECT
    date,
    item_id,
    store_code,
    'store'::VARCHAR AS scope,
    cate_id,
    cate_level_id,
    brand_id,
    supplier_id,
    qty_alipay_njhs AS demand,
    pv_ipv,
    pv_uv,
    cart_ipv,
    cart_uv,
    collect_uv,
    num_gmv,
    qty_gmv,
    unum_gmv,
    num_alipay,
    qty_alipay,
    unum_alipay,
    ztc_pv_uv,
    tbk_pv_uv,
    ss_pv_uv,
    jhs_pv_uv,
    num_alipay_njhs,
    qty_alipay_njhs,
    unum_alipay_njhs
FROM clean_item_store_feature;
