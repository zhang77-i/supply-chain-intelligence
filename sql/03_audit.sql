-- Data quality audit for forecasting pipeline

SELECT
    COUNT(*) AS total_rows,
    COUNT(*) - COUNT(item_id) AS missing_item_id,
    COUNT(DISTINCT item_id) AS unique_items
FROM daily_panel;

-- Check duplicated SKU-store-date records
SELECT
    item_id,
    store_code,
    date,
    COUNT(*) AS cnt
FROM daily_panel
GROUP BY item_id, store_code, date
HAVING COUNT(*) > 1;
