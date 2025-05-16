Quantitative Strategy Backtesting Engine

Overview

This engine is a modular, database-integrated framework built for institutional-grade backtesting of quantitative trading strategies. It supports realistic trading constraints like slippage, commissions, and dynamic capital allocation, while enabling detailed trade event logging, symbol scoring, and performance attribution through the Fama-French 3-Factor model.

⸻

Architecture Summary

1. Strategy Execution
	•	The tradebook() function executes any user-defined strategy (e.g., mean_reversion) across all symbols.
	•	It pulls OHLC data from the nsedata_fact table and generates trade signals with entry/exit timestamps, quantity, and stop-loss logic.

2. Trade Logging
	•	All trades are recorded in the trade_book PostgreSQL table.
	•	Each record includes:
	•	Symbol
	•	Signal (Buy/Sell)
	•	Entry & Exit Price
	•	Entry & Exit Time
	•	Quantity
	•	P&L
	•	Stop-loss and execution slippage

3. Data Integration
	•	Price and fundamental data are pulled from the nsedata_fact table.
	•	All indicator computation (EMA, RSI, ADX, DI) is done in-memory using Pandas, NumPy, and technical analysis libraries.

4. Metrics Calculation
	•	Performance metrics such as Sharpe Ratio, CAGR, Win Rate, Max Drawdown, Annual Volatility, Profit Factor, and Value at Risk are calculated daily.
	•	Results are benchmarked against market indices like NIFTY 50.

5. Symbol Scoring
	•	A scoring mechanism ranks each symbol daily using normalized metrics (e.g., Sharpe, win rate, and drawdown).
	•	These scores drive capital allocation decisions.

6. Capital Allocation
	•	Capital is allocated daily based on:
	•	initial_capital
	•	capital_exposure (% of capital available for allocation)
	•	buffer_pct (capital reserved to avoid full utilization)
	•	max_risk (risk per trade as a % of capital)
	•	Partial fills are not modeled, slippage and commission are accounted for realistically.

7. Risk Management
	•	Stop-loss is calculated using ATR or custom logic, ensuring no trade violates the risk budget.
	•	Quantity per trade is dynamically adjusted based on stop distance and risk budget.
	•	Capital preservation logic ensures drawdowns are contained.

8. Execution Simulation
	•	Slippage is applied on every trade to simulate realistic fills.
	•	Commission costs are subtracted to reflect net profitability.
	•	Execution supports partial fills and position rebalancing.

9. Portfolio Management
	•	Portfolio equity is tracked daily using MTM valuations.
	•	Trade-level cash flows are aggregated to compute equity curve and unallocated reserves.
	•	Unused capital is preserved for higher-scoring opportunities.

10. Alpha Attribution
	•	Fama-French 3-Factor model is used to run a regression on daily returns.
	•	Output includes:
	•	Alpha
	•	Market Beta
	•	SMB/HML Betas
	•	R²
	•	p-values

--

Requirements

Ensure the following are installed:
	•	Python >= 3.8
	•	PostgreSQL with:
	•	trade_book table
	•	nsedata_fact table (for historical data)
	•	Python Packages:
	•	pandas
	•	numpy
	•	psycopg2
	•	scikit-learn
	•	statsmodels
	•	yfinance (if applicable for fetching data)
 

<img width="448" alt="Screenshot 2025-05-16 at 1 27 12 PM" src="https://github.com/user-attachments/assets/f26784bb-3177-4aef-82c5-8a419c1c93d7" />

Sample results:

	1. Strategy performance - 
 		CAGR: 19.22%
		Sharpe Ratio: 0.6073
		Max Drawdown: -28.4%
		Win Rate: 52.78%
		Profit Factor: 1.1242
		Final Portfolio Value: ₹15,11,575.96
  	2, Index Comparison (NIFTY 50) - 
		CAGR: 20.86%
		Sharpe Ratio: 0.4681
		Max Drawdown: -15.77%
		Annual Volatility: 12.77%
  	3. Fama-French Attribution - 
   		Alpha: 0.00160527
		Market Beta: 0.17584
		SMB Beta: -0.39188
		HML Beta: 0.05935
		R²: 0.00377


Author:

Dhruv Khatri
dhruvkhatri9275@gmail.com