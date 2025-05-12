CREATE TABLE portfolio_log as (
WITH active_holdings AS (
    SELECT
        symbol,
        entry_date,
        exit_date,
        quantity
    FROM closed_trades
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
    FROM active_holdings h
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
        SUM(entry_price * quantity) AS total_investment,
		SUM(total_invested + total_commission) as total_cost
    FROM closed_trades
    GROUP BY entry_date
)
SELECT
    v.date,
    v.portfolio_value,
    COALESCE(i.total_investment, 0) AS total_investment,
	COALESCE(i.total_cost, 0) AS total_cost
FROM daily_valuation v
LEFT JOIN daily_investment i ON v.date = i.date
ORDER BY v.date);