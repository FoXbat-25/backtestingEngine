from backtestingEngine.app.backtest_XX import fetch_nse_data, daily_returns

def main():

    df = fetch_nse_data('TATAMOTORS')
    daily_returns(df)

if __name__ == "__main__":
    main()
