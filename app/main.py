from engine import *
from meanReversion.app.mean_reversion import mean_reversion
from dynamic_allocation import dynamic_allocation, trade_log_insertion, get_portfolio_log
from utils import backTester
from utils import *
import matplotlib.pyplot as plt

def main():
    nATR = 2.5 #Used to guage stop loss
    strategy = "MeanReversion"
    # symbol = "ICICIBANK"
    initial_capital = 1000000
    capital_exposure = 0.4
    commission = 0.0005
    max_risk = 0.01
    risk_free_rate = 0.07
    # confidence_level = 0.95
    start_date = '2023-01-01'
    slippage_rate = 0.001
    buffer_pct = 0.25
    
    backtest_obj = backTester(nATR, strategy_func=mean_reversion, strategy_name=strategy, start_from=start_date)
    df = backtest_obj.trades_df
    
    df = slippage(df, slippage_rate)
    metrics_df = get_daily_metrics(risk_free_rate)
    df=df.merge(metrics_df[['symbol','score']], on='symbol', how='left')
    event_df = get_indv_trades(df)
    # event_df.to_csv('event_df.csv')
    trade_log = dynamic_allocation(event_df,strategy, initial_capital, capital_exposure, buffer_pct, max_risk, commission=commission)
    # print(trade_log)
    trade_log_insertion(trade_log)
    portfolio_df = get_portfolio_log(initial_capital)
    portfolio_df.to_csv('portfolio.csv')
    french_fama_results = french_fama_three(portfolio_df)
    results = get_portfolio_metrics(portfolio_df, initial_capital)
    print(results)
    index_results = get_index_metrics(start_date, risk_free_rate)
    print(index_results)
    print(f'French fama = {french_fama_results}')
    # plt.plot(portfolio_df['date'], portfolio_df['asset_valuation'] )
    print('operations complete')
    
if __name__ == "__main__":
    main()
