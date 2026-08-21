-- Create leakage-safe forecasting features
-- Features must be generated only from information available before forecast cutoff

CREATE TABLE feature_snapshot AS
SELECT
    item_id,
    store_code,
    date,
    AVG(demand) OVER(
        PARTITION BY item_id
        ORDER BY date
        ROWS BETWEEN 27 PRECEDING AND CURRENT ROW
    ) AS ma28
FROM daily_panel;
