CREATE OR REPLACE VIEW raw_item_feature AS
SELECT *
FROM read_csv(
    '{{ITEM_FEATURE_PATH}}',
    header = false,
    auto_detect = false,
    columns = {{ITEM_COLUMN_MAP}},
    delim = ',',
    quote = '"',
    strict_mode = true
);

CREATE OR REPLACE VIEW raw_item_store_feature AS
SELECT *
FROM read_csv(
    '{{ITEM_STORE_FEATURE_PATH}}',
    header = false,
    auto_detect = false,
    columns = {{STORE_COLUMN_MAP}},
    delim = ',',
    quote = '"',
    strict_mode = true
);

CREATE OR REPLACE VIEW raw_config AS
SELECT *
FROM read_csv(
    '{{CONFIG_PATH}}',
    header = false,
    auto_detect = false,
    columns = {{CONFIG_COLUMN_MAP}},
    delim = ',',
    quote = '"',
    strict_mode = true
);

CREATE OR REPLACE VIEW raw_sample_submission AS
SELECT
    item_id,
    store_code,
    target
FROM read_csv(
    '{{SAMPLE_SUBMISSION_PATH}}',
    header = false,
    auto_detect = false,
    columns = {'item_id': 'BIGINT', 'store_code': 'VARCHAR', 'target': 'DOUBLE'},
    delim = ',',
    strict_mode = true
);
