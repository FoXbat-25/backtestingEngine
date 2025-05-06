from engine import *
from meanReversion.app.mean_reversion import mean_reversion
import pprint

def main():
    strategy = "MeanReversion"
    symbol = "KPIL"
    initial_capital = 1000000
    
    # backTester(strategy_func=mean_reversion, strategy_name=strategy) # This inserts data into postgresql

    symbols_df = fetch_symbols()
    results_list = []

    symbols_df = symbols_df['symbol'].unique()
    for symbol in symbols_df:   

        trade_df = order_book_transformation(symbol, strategy, initial_capital)

        if not trade_df.empty:
            results = metrics(trade_df, initial_capital)
            results_list.append(results)
        
        else:
            print(f'No trade for symbol: {symbol}.')
        
    final_df = pd.concat(results_list, ignore_index=True)

    final_df = final_df.sort_values(by='sharpe_ratio', ascending=False)
    # final_df.to_csv("output.csv", index=False)
    print(final_df)


    # trade_df = order_book_transformation(symbol, strategy, initial_capital)
    # # pd.set_option('display.max_columns', None)
    # print(trade_df)



if __name__ == "__main__":
    main()




# df = fetch_data(symbol, start_date='2023-01-01')
# daily_df = daily_returns(df)

# for key, value in daily_df.items():
#     print(f"{key}: {value}")