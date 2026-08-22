CREATE OR REPLACE TABLE dim_series AS
WITH observed AS (
    SELECT
        item_id,
        store_code,
        MIN(date) AS first_observed_date,
        MAX(date) AS last_observed_date,
        COUNT(*) AS observed_rows
    FROM fact_demand
    GROUP BY item_id, store_code
)
SELECT
    c.item_id,
    c.store_code,
    CASE WHEN c.store_code = 'all' THEN 'national' ELSE 'store' END AS scope,
    c.shortage_cost,
    c.overage_cost,
    c.valid_cost_pair,
    c.critical_fractile,
    o.first_observed_date,
    o.last_observed_date,
    o.observed_rows
FROM clean_config c
LEFT JOIN observed o
  ON c.item_id = o.item_id
 AND c.store_code = o.store_code;

CREATE OR REPLACE TABLE calendar AS
SELECT CAST(day AS DATE) AS date
FROM GENERATE_SERIES(
    (SELECT MIN(date) FROM fact_demand),
    (SELECT MAX(date) FROM fact_demand),
    INTERVAL 1 DAY
) AS generated(day);

CREATE OR REPLACE TABLE daily_panel AS
SELECT
    calendar.date,
    series.item_id,
    series.store_code,
    series.scope,
    item.cate_id,
    item.cate_level_id,
    item.brand_id,
    item.supplier_id,
    series.shortage_cost,
    series.overage_cost,
    series.critical_fractile,
    series.first_observed_date,
    series.last_observed_date,
    DATE_DIFF('day', series.first_observed_date, calendar.date) AS days_since_first_observed,
    fact.item_id IS NOT NULL AS is_observed,
    calendar.date > series.last_observed_date AS is_after_last_observed,
    GREATEST(COALESCE(fact.demand, 0), 0) AS demand,
    GREATEST(COALESCE(fact.pv_ipv, 0), 0) AS pv_ipv,
    GREATEST(COALESCE(fact.pv_uv, 0), 0) AS pv_uv,
    GREATEST(COALESCE(fact.cart_ipv, 0), 0) AS cart_ipv,
    GREATEST(COALESCE(fact.cart_uv, 0), 0) AS cart_uv,
    GREATEST(COALESCE(fact.collect_uv, 0), 0) AS collect_uv,
    GREATEST(COALESCE(fact.num_gmv, 0), 0) AS num_gmv,
    GREATEST(COALESCE(fact.qty_gmv, 0), 0) AS qty_gmv,
    GREATEST(COALESCE(fact.unum_gmv, 0), 0) AS unum_gmv,
    GREATEST(COALESCE(fact.num_alipay, 0), 0) AS num_alipay,
    GREATEST(COALESCE(fact.qty_alipay, 0), 0) AS qty_alipay,
    GREATEST(COALESCE(fact.unum_alipay, 0), 0) AS unum_alipay,
    GREATEST(COALESCE(fact.ztc_pv_uv, 0), 0) AS ztc_pv_uv,
    GREATEST(COALESCE(fact.tbk_pv_uv, 0), 0) AS tbk_pv_uv,
    GREATEST(COALESCE(fact.ss_pv_uv, 0), 0) AS ss_pv_uv,
    GREATEST(COALESCE(fact.jhs_pv_uv, 0), 0) AS jhs_pv_uv,
    GREATEST(COALESCE(fact.num_alipay_njhs, 0), 0) AS num_alipay_njhs,
    GREATEST(COALESCE(fact.qty_alipay_njhs, 0), 0) AS qty_alipay_njhs,
    GREATEST(COALESCE(fact.unum_alipay_njhs, 0), 0) AS unum_alipay_njhs
FROM dim_series series
JOIN calendar
  ON series.first_observed_date IS NOT NULL
 AND calendar.date BETWEEN series.first_observed_date
                       AND (SELECT MAX(date) FROM fact_demand)
LEFT JOIN fact_demand fact
  ON fact.item_id = series.item_id
 AND fact.store_code = series.store_code
 AND fact.date = calendar.date
LEFT JOIN dim_item item
  ON item.item_id = series.item_id;

CREATE INDEX IF NOT EXISTS idx_daily_panel_series_date
ON daily_panel(item_id, store_code, date);
