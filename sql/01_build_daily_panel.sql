-- Build SKU daily demand panel
-- Purpose: transform transactional demand into forecasting-ready time series

CREATE TABLE daily_panel AS
SELECT
    item_id,
    store_code,
    date,
    SUM(demand) AS demand
FROM clean_sales
GROUP BY item_id, store_code, date;
