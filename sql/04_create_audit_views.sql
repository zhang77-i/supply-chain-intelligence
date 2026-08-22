CREATE OR REPLACE VIEW national_store_daily_consistency AS
WITH national AS (
    SELECT date, item_id, SUM(qty_alipay_njhs) AS national_demand
    FROM clean_item_feature
    GROUP BY date, item_id
),
stores AS (
    SELECT date, item_id, SUM(qty_alipay_njhs) AS store_demand
    FROM clean_item_store_feature
    GROUP BY date, item_id
)
SELECT
    COALESCE(national.date, stores.date) AS date,
    COALESCE(national.item_id, stores.item_id) AS item_id,
    COALESCE(national.national_demand, 0) AS national_demand,
    COALESCE(stores.store_demand, 0) AS store_demand,
    COALESCE(national.national_demand, 0) - COALESCE(stores.store_demand, 0) AS demand_diff
FROM national
FULL OUTER JOIN stores
  ON national.date = stores.date
 AND national.item_id = stores.item_id;

CREATE OR REPLACE VIEW config_coverage AS
SELECT
    series.item_id,
    series.store_code,
    series.valid_cost_pair,
    series.first_observed_date IS NOT NULL AS has_demand_history
FROM dim_series series;
