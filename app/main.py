from engine import *
from meanReversion.app.mean_reversion import mean_reversion
from dynamic_allocation import dynamic_allocation
from utils import *

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
    
    metrics_df, all_orders_df  = all_orders_and_metrics(strategy, initial_capital)
    daily_metrics_df = all_daily_metrics(risk_free_rate)
    all_orders_df=all_orders_df.merge(daily_metrics_df[['symbol','score']], on='symbol', how='left')
    trade_log = dynamic_allocation(all_orders_df, initial_capital, capital_exposure, max_risk, commission=commission)
    trade_log = trade_log.sort_values('date', ascending= True)
    print(trade_log)

if __name__ == "__main__":
    main()



# df = order_book_transformation(symbol, strategy, initial_capital)
    # print(all_orders_df)
    # pd.set_option('display.max_columns', None)