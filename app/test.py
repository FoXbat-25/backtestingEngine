from engine import *
from meanReversion.app.mean_reversion import mean_reversion
import pprint

def main():
    nATR = 2.5 #Used to guage stop loss
    strategy = "MeanReversion"
    symbol = "ICICIBANK"
    initial_capital = 1000000
    
    # backTester(nATR, strategy_func=mean_reversion, strategy_name=strategy) # This inserts data into postgresql
    df = order_book_transformation(symbol, strategy, initial_capital)
    # metrics_df, orders_df  = all_orders_and_metrics(strategy, initial_capital)
    # metrics_df['symbol'] = metrics_df['symbol']
    # orders_df=orders_df.merge(metrics_df[['symbol','score']], on='symbol', how='left')
    # pd.set_option('display.max_columns', None)
    print(df)
    # orders_df=orders_df.sort_values(by='entry_date', ascending = True)
    #pd.set_option('display.max_columns', None)
    # print(orders_df)
    event_df = indv_trade_listing(df)
    event_df = event_df.sort_values(by='date')
    print(event_df)
if __name__ == "__main__":
    main()




# df = fetch_data(symbol, start_date='2023-01-01')
# daily_df = daily_returns(df)

# for key, value in daily_df.items():
#     print(f"{key}: {value}")