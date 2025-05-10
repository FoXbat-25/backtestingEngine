from engine import *
from meanReversion.app.mean_reversion import mean_reversion
import pprint

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
    
    # backTester(nATR, strategy_func=mean_reversion, strategy_name=strategy) # This inserts data into postgresql
    # df = order_book_transformation(symbol, strategy, initial_capital)
    metrics_df, all_orders_df  = all_orders_and_metrics(strategy, initial_capital)
    daily_metrics_df = all_daily_metrics(risk_free_rate)
    all_orders_df=all_orders_df.merge(daily_metrics_df[['symbol','score']], on='symbol', how='left')
    # print(all_orders_df)
    # pd.set_option('display.max_columns', None)
    trade_log = dynamic_allocation(all_orders_df, initial_capital, capital_exposure, max_risk, commission=commission)
    trade_log = trade_log.sort_values('date', ascending= True)
    print(trade_log)
if __name__ == "__main__":
    main()




# df = fetch_data(symbol, start_date='2023-01-01')
# daily_df = daily_returns(df)

# for key, value in daily_df.items():
#     print(f"{key}: {value}")