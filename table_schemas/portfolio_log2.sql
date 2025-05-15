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
),
order_summary as (
SELECT
    b.symbol,
    b.entry_date,
    b.entry_price,
    s.exit_date,
    s.exit_price,
    b.quantity,
    b.quantity * b.entry_price AS total_invested,
	b.quantity * s.exit_price AS total_sold,
    b.commision_cost AS buy_commission,
	s.commision_cost AS sell_commision,
	b.commision_cost + s.commision_cost AS total_commision
FROM buys b
JOIN sells s
  ON b.symbol = s.symbol AND b.rn = s.rn
ORDER BY b.symbol, b.entry_date
),
date_range AS (
    SELECT DISTINCT date
    FROM nsedata_fact WHERE date >= '2023-01-01'
),
holding_days AS (
    SELECT
        d.date,
        h.symbol,
        h.quantity
    FROM order_summary h
    JOIN date_range d
      ON d.date BETWEEN h.entry_date AND h.exit_date
),
valuation AS (
    SELECT
        hd.date,
        hd.symbol,
        hd.quantity,
        nf.close,
        hd.quantity * nf.close AS holding_value
    FROM holding_days hd
    JOIN nsedata_fact nf
      ON hd.symbol = nf.symbol AND hd.date = nf.date
),
daily_valuation AS (
	SELECT
	    date,
	    SUM(holding_value) AS portfolio_value
	FROM valuation
	GROUP BY date
),
daily_investment AS (
    SELECT
        entry_date AS date,
        --SUM(entry_price * quantity) AS total_investment,
		SUM(total_invested + buy_commission) as total_buy_cost
    FROM order_summary
    GROUP BY entry_date
),
daily_sold AS (
	SELECT 
		exit_date AS date,
		SUM(exit_price * quantity) AS total_exit,
		SUM(total_sold - sell_commision) AS total_sell_value
	FROM order_summary
	GROUP BY exit_date
)
SELECT
    v.date,
    v.portfolio_value,
    --COALESCE(i.total_investment, 0) AS total_investment,
	COALESCE(i.total_buy_cost, 0) AS total_buy_cost,
	COALESCE(m.total_sell_value, 0) AS total_sell_value
FROM daily_valuation v
LEFT JOIN daily_investment i ON v.date = i.date
LEFT JOIN daily_sold m ON v.date = m.date
ORDER BY v.date;