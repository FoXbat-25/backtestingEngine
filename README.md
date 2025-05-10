+-------------------------------+
|    tradebook(strategy_fn)     |
+-------------------------------+
             ↓
+-------------------------------+
|    Signal generation for N    |
|     symbols                    |
+-------------------------------+
             ↓
+-------------------------------+
|    Log trade in trade_book    |
|    (PostgreSQL)               |
+-------------------------------+
             ↓
+-------------------------------+
|   Fetch OHLC data from        |
|   nsedata_fact                |
+-------------------------------+
             ↓
+-------------------------------+
|   Calculate daily metrics     |
|   (Sharpe, WinRate, etc.)     |
+-------------------------------+
             ↓
+-------------------------------+
|   Symbol score calculation    |
|   and rank symbols            |
+-------------------------------+
             ↓
+-------------------------------+
|   Capital allocation based on |
|   score and risk per trade    |
+-------------------------------+
             ↓
+-------------------------------+
|   Execute trades with slippage|
|   and commission applied      |
+-------------------------------+
             ↓
+-------------------------------+
|   Track portfolio balance     |
|   (including reserves)        |
+-------------------------------+

  1.	Strategy Execution:
The tradebook() function takes the selected strategy function as an argument and generates trade signals for the specified symbols.
	2.	Trade Logging:
All trade details are logged into the trade_book PostgreSQL table, including signal data, entry/exit prices, and P&L calculations.
	3.	Data Integration:
OHLC data is fetched from the nsedata_fact table, which contains daily prices for each symbol. This data is used for backtesting strategy performance.
	4.	Metrics Calculation:
The backtest calculates performance metrics (Sharpe ratio, win rate, drawdown, etc.) for each symbol on a daily basis.
	5.	Symbol Scoring:
Each symbol is assigned a score based on its daily metrics, helping to rank it in terms of profitability and risk. The higher the score, the higher the allocation.
	6.	Capital Allocation:
A dynamic allocation model is employed, where capital exposure is determined by the exposure_cap. This ensures that risk per trade is managed, and a cash reserve is maintained.
	7.	Risk Management:
The stop-loss logic ensures that no position exceeds a predefined risk threshold. The quantity for each trade is calculated based on the risk per trade and stop-loss distance.
	8.	Execution:
Trades are executed with slippage and commission taken into account. The execution logic includes partial fills for more realistic trade simulation.
	9.	Portfolio Management:
Throughout the backtest, portfolio balance and allocated capital are tracked to ensure consistent capital use across trades. The system also accounts for any unallocated reserves for future opportunities.

Ensure that the following tools are installed:
	•	Python 3.8+
	•	PostgreSQL (for trade_book and nsedata_fact)
	•	Required Python packages (yfinance, pandas, numpy, psycopg2, etc.)


To get started - 
 git clone https://github.com/FoXbat-25/backtestingEngine.git
 cd nseDaily

Use the backtester class only once, it inserts data into DB, comment after first run. 

 
<img width="1023" alt="Screenshot 2025-05-10 at 9 35 50 PM" src="https://github.com/user-attachments/assets/e635f126-31ba-4308-a579-289f67321166" />
