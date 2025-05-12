WITH buys AS (
    SELECT
        symbol,
        date AS entry_date,
        price_adj AS entry_price,
        quantity,
        total_cost,
        commision_cost,
        ROW_NUMBER() OVER (PARTITION BY symbol ORDER BY date) AS rn
    FROM order_logs
    WHERE order_type = 'Buy'
),
sells AS (
    SELECT
        symbol,
        date AS exit_date,
        price_adj AS exit_price,
        quantity,
        total_cost,
        commision_cost,
        ROW_NUMBER() OVER (PARTITION BY symbol ORDER BY date) AS rn
    FROM order_logs
    WHERE order_type = 'Sell'
)
SELECT
    b.symbol,
    b.entry_date,
    b.entry_price,
    s.exit_date,
    s.exit_price,
    b.quantity,
    b.quantity * b.entry_price AS total_invested,
    b.commision_cost + s.commision_cost AS total_commission
INTO order_summary
FROM buys b
JOIN sells s
  ON b.symbol = s.symbol AND b.rn = s.rn
ORDER BY b.symbol, b.entry_date;

-- the trade_log dataframe has been transformed into a summary table with entry and exit dates