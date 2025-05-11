import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from engine import *
from meanReversion.app.mean_reversion import mean_reversion
from dynamic_allocation import dynamic_allocation
import pprint
from engine2 import *
from trade_book import trade_book2

def main():
    nATR = 2.5 #Used to guage stop loss
    strategy = "MeanReversion"
    symbol = "ICICIBANK"
    initial_capital = 1000000
    capital_exposure = 0.4
    commission = 0.0005
    max_risk = 0.01
    risk_free_rate = 0.07
    confidence_level = 0.95
    start_date = '2023-01-01'
    slippage_rate = 0.001
    
    backtest_obj = backTester(nATR, strategy_func=mean_reversion, strategy_name=strategy) # This inserts data into postgresql
    df = backtest_obj.trades_df

    df = slippage(df, slippage_rate)
    metrics_df = get_daily_metrics(risk_free_rate)
    df=df.merge(metrics_df[['symbol','score']], on='symbol', how='left')
    event_df = get_indv_trades(df)
    trade_log = dynamic_allocation(event_df, initial_capital, capital_exposure, max_risk, commission=commission)
    print(trade_log)

    # df = mean_reversion(start_from='2023-01-01')
    # trades_df = trade_book2(nATR, df, strategy)
    # print(trades_df)
    # df.to_csv('~/Desktop/trades.csv')
    # trade_log = dynamic_allocation(df, initial_capital, capital_exposure, max_risk, commission=commission)
    # trade_log = trade_log.sort_values('date', ascending= True)
    # print(trade_log)
    # trade_log.to_csv('trade_log.csv')
    
    # df=df.merge(metrics_df[['symbol','score']], on='symbol', how='left')
    # event_df = indv_trade_listing(df)
    # event_df = event_df.sort_values(by='date')
    # df = dynamic_allocation(event_df, initial_capital, capital_exposure, max_risk, commission=commission)
    
    # df = fetch_data(symbol)
    # daily_df = daily_metrics(df,risk_free_rate)
    # print(daily_df)

    # final_df = all_daily_metrics(risk_free_rate)
    # print(final_df)
if __name__ == "__main__":
    main()




# df = fetch_data(symbol, start_date='2023-01-01')
# daily_df = daily_returns(df)

# for key, value in daily_df.items():
#     print(f"{key}: {value}")