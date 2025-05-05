from engine import *
from meanReversion.app.mean_reversion import mean_reversion
import pprint

def main():
    strategy = "MeanReversion"
    symbol = "TATAMOTORS"
    initial_capital = 300000
    
    # backTester(strategy_func=mean_reversion, strategy_name=strategy)

    df = fetch_data(symbol)
    data = daily_returns(df)
    pprint.pprint(data)

    # trade_df = order_book_transformation(symbol, strategy) 
    # results = metrics(trade_df, initial_capital)

    # print(results)

    # print("mean daily return", data["mean daily return"])
    # print("annual return", data["annual return"])
    # print("std_dev_daily_return", data["std_dev_daily_return"])
    # print("annual_std_dev", data["annual_std_dev"])
    # print("buy_and_hold_return", data["buy_and_hold_return"])

if __name__ == "__main__":
    main()
