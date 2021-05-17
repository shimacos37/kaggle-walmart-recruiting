
DECLARE START_TS DATE DEFAULT DATE(TIMESTAMP("{{start_ts}}"));
DECLARE END_TS DATE DEFAULT DATE(TIMESTAMP("{{end_ts}}"));

WITH BASE_DATA AS (
  SELECT
    Store,
    Date AS start_date,
    UNIX_SECONDS(TIMESTAMP(Date)) AS unixtime,
    Dept,
    IsHoliday,
    Weekly_Sales,
    Type,
    Size,
    Temperature,
    Fuel_Price,
    {% for i in range(1, 6) %}
    CASE 
      WHEN MarkDown{{i}} = 'NA' THEN Null
      ELSE CAST(MarkDown{{i}} AS FLOAT64)
    END AS MarkDown{{i}},
    {% endfor %}
    CASE 
      WHEN CPI = 'NA' THEN Null
      ELSE CAST(CPI AS FLOAT64)
    END AS CPI,
    CASE 
      WHEN Unemployment = 'NA' THEN Null
      ELSE CAST(Unemployment AS FLOAT64)
    END AS Unemployment,
  FROM (
    SELECT Store, Dept, Date, Weekly_Sales FROM walmart.train
    UNION ALL
    -- test部分はnullでtargetの値を埋めるがWINDOW関数でこの部分にはアクセスしないので問題ない
    SELECT Store, Dept, Date, Null AS Weekly_Sales FROM walmart.test
    WHERE Date <= END_TS
  )
  LEFT JOIN walmart.stores USING (Store)
  LEFT JOIN walmart.features USING (Store, Date)
),
TARGET_FEATURE AS (
  SELECT
    Store,
    Dept,
    start_date,
    {% for groupby_col, groupby_name in groupby_cols %}
      {% for op, op_name in aggregate_ops %}
          -- 半年間分の集計特徴量
          {{op}}(Weekly_Sales) OVER (
              PARTITION BY {{groupby_col}} ORDER BY unixtime 
              RANGE BETWEEN {{half_year_secs + base_secs}} PRECEDING AND {{base_secs}} PRECEDING
          ) AS sales_{{op_name}}_by_{{groupby_name}}_half_year,
      {% endfor %}
      {% for op, op_name in aggregate_ops %}
          -- 2ヶ月分の集計特徴量
          {{op}}(Weekly_Sales) OVER (
              PARTITION BY {{groupby_col}} ORDER BY unixtime 
              RANGE BETWEEN {{two_month_secs + base_secs}} PRECEDING AND {{base_secs}} PRECEDING
          ) AS sales_{{op_name}}_by_{{groupby_name}}_two_month,
      {% endfor %}
      {% for op, op_name in aggregate_ops %}
          -- 1ヶ月分の集計特徴量
          {{op}}(Weekly_Sales) OVER (
              PARTITION BY {{groupby_col}} ORDER BY unixtime 
              RANGE BETWEEN {{one_month_secs + base_secs}} PRECEDING AND {{base_secs}} PRECEDING
          ) AS sales_{{op_name}}_by_{{groupby_name}}_one_month,
      {% endfor %}
    {% endfor %}
  FROM
    BASE_DATA
)
SELECT
  *
FROM
  BASE_DATA
LEFT JOIN TARGET_FEATURE USING(Store, Dept, start_date)
WHERE
  # 最大で半年分の集計を取ってるため、最初の数ヶ月は訓練からは外す
  start_date >= START_TS

