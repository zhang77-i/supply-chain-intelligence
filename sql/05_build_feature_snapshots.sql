CREATE OR REPLACE TABLE feature_snapshots_raw AS
WITH windowed AS (
    SELECT
        panel.*,
        ROW_NUMBER() OVER series_window AS history_days_total,
        LAG(demand, 1) OVER series_window AS demand_lag_1,
        LAG(demand, 7) OVER series_window AS demand_lag_7,
        LAG(demand, 14) OVER series_window AS demand_lag_14,
        LAG(demand, 28) OVER series_window AS demand_lag_28,

        COUNT(*) OVER window_84 AS history_days_84,
        SUM(CASE WHEN is_observed THEN 1 ELSE 0 END) OVER window_84 AS observed_days_84,
        SUM(CASE WHEN demand > 0 THEN 1 ELSE 0 END) OVER window_84 AS nonzero_days_84,
        AVG(CASE WHEN demand > 0 THEN demand END) OVER window_84 AS nonzero_demand_mean_84,
        STDDEV_SAMP(CASE WHEN demand > 0 THEN demand END) OVER window_84 AS nonzero_demand_std_84,

        SUM(demand) OVER window_7 AS demand_sum_7,
        AVG(demand) OVER window_7 AS demand_mean_7,
        STDDEV_SAMP(demand) OVER window_7 AS demand_std_7,
        SUM(demand) OVER window_14 AS demand_sum_14,
        AVG(demand) OVER window_14 AS demand_mean_14,
        STDDEV_SAMP(demand) OVER window_14 AS demand_std_14,
        SUM(demand) OVER window_28 AS demand_sum_28,
        AVG(demand) OVER window_28 AS demand_mean_28,
        STDDEV_SAMP(demand) OVER window_28 AS demand_std_28,
        SUM(demand) OVER window_56 AS demand_sum_56,
        AVG(demand) OVER window_56 AS demand_mean_56,

        SUM(pv_uv) OVER window_14 AS pv_uv_sum_14,
        SUM(cart_uv) OVER window_14 AS cart_uv_sum_14,
        SUM(collect_uv) OVER window_14 AS collect_uv_sum_14,
        SUM(qty_gmv) OVER window_14 AS qty_gmv_sum_14,
        SUM(qty_alipay) OVER window_14 AS qty_alipay_sum_14,
        SUM(unum_alipay_njhs) OVER window_14 AS pay_uv_sum_14,
        SUM(ztc_pv_uv) OVER window_14 AS ztc_pv_uv_sum_14,
        SUM(tbk_pv_uv) OVER window_14 AS tbk_pv_uv_sum_14,
        SUM(ss_pv_uv) OVER window_14 AS ss_pv_uv_sum_14,
        SUM(jhs_pv_uv) OVER window_14 AS jhs_pv_uv_sum_14,

        SUM(pv_uv) OVER window_28 AS pv_uv_sum_28,
        SUM(cart_uv) OVER window_28 AS cart_uv_sum_28,
        SUM(qty_alipay_njhs) OVER window_28 AS qty_njhs_sum_28,

        SUM(demand) OVER future_14 AS target_14d_unchecked,
        COUNT(*) OVER future_14 AS future_days_available
    FROM daily_panel panel
    WINDOW
        series_window AS (
            PARTITION BY item_id, store_code
            ORDER BY date
        ),
        window_7 AS (
            PARTITION BY item_id, store_code
            ORDER BY date ROWS BETWEEN 6 PRECEDING AND CURRENT ROW
        ),
        window_14 AS (
            PARTITION BY item_id, store_code
            ORDER BY date ROWS BETWEEN 13 PRECEDING AND CURRENT ROW
        ),
        window_28 AS (
            PARTITION BY item_id, store_code
            ORDER BY date ROWS BETWEEN 27 PRECEDING AND CURRENT ROW
        ),
        window_56 AS (
            PARTITION BY item_id, store_code
            ORDER BY date ROWS BETWEEN 55 PRECEDING AND CURRENT ROW
        ),
        window_84 AS (
            PARTITION BY item_id, store_code
            ORDER BY date ROWS BETWEEN {{HISTORY_PRECEDING}} PRECEDING AND CURRENT ROW
        ),
        future_14 AS (
            PARTITION BY item_id, store_code
            ORDER BY date ROWS BETWEEN {{HORIZON_FOLLOWING_START}} FOLLOWING
                                   AND {{HORIZON_DAYS}} FOLLOWING
        )
)
SELECT
    windowed.date AS cutoff_date,
    cutoffs.cutoff_kind,
    windowed.item_id,
    windowed.store_code,
    windowed.scope,
    windowed.cate_id,
    windowed.cate_level_id,
    windowed.brand_id,
    windowed.supplier_id,
    windowed.shortage_cost,
    windowed.overage_cost,
    windowed.critical_fractile,
    windowed.days_since_first_observed,
    windowed.is_observed,
    windowed.is_after_last_observed,
    windowed.demand AS demand_today,
    windowed.demand_lag_1,
    windowed.demand_lag_7,
    windowed.demand_lag_14,
    windowed.demand_lag_28,
    windowed.history_days_total,
    windowed.history_days_84,
    windowed.observed_days_84,
    windowed.nonzero_days_84,
    windowed.nonzero_demand_mean_84,
    windowed.nonzero_demand_std_84,
    windowed.demand_sum_7,
    windowed.demand_mean_7,
    windowed.demand_std_7,
    windowed.demand_sum_14,
    windowed.demand_mean_14,
    windowed.demand_std_14,
    windowed.demand_sum_28,
    windowed.demand_mean_28,
    windowed.demand_std_28,
    windowed.demand_sum_56,
    windowed.demand_mean_56,
    windowed.pv_uv_sum_14,
    windowed.cart_uv_sum_14,
    windowed.collect_uv_sum_14,
    windowed.qty_gmv_sum_14,
    windowed.qty_alipay_sum_14,
    windowed.pay_uv_sum_14,
    windowed.ztc_pv_uv_sum_14,
    windowed.tbk_pv_uv_sum_14,
    windowed.ss_pv_uv_sum_14,
    windowed.jhs_pv_uv_sum_14,
    windowed.pv_uv_sum_28,
    windowed.cart_uv_sum_28,
    windowed.qty_njhs_sum_28,
    windowed.future_days_available,
    CASE
        WHEN cutoffs.cutoff_kind = 'backtest'
         AND windowed.future_days_available = {{HORIZON_DAYS}}
        THEN windowed.target_14d_unchecked
        ELSE NULL
    END AS target_14d
FROM windowed
INNER JOIN model_cutoffs cutoffs
  ON windowed.date = cutoffs.cutoff_date;
