from backtestingEngine.app.engine import *
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
    start_date = '2023-01-01'
    slippage_rate = 0.001
    
    backtest_obj = backTester(nATR, strategy_func=mean_reversion, strategy_name=strategy, start_from=start_date)
    df = backtest_obj.trades_df

    df = slippage(df, slippage_rate)
    metrics_df = get_daily_metrics(risk_free_rate)
    df=df.merge(metrics_df[['symbol','score']], on='symbol', how='left')
    event_df = get_indv_trades(df)
    trade_log = dynamic_allocation(event_df, initial_capital, capital_exposure, max_risk, commission=commission)
    print(trade_log)
    
if __name__ == "__main__":
    main()



# df = order_book_transformation(symbol, strategy, initial_capital)
    # print(all_orders_df)
    # pd.set_option('display.max_columns', None)