import pandas as pd
import numpy as np
import matplotlib as plt
from config import SQL_ALCHEMY_CONN
from sqlalchemy import create_engine

from  datetime import datetime

from trade_book_copy import trade_book

engine = create_engine(SQL_ALCHEMY_CONN)

def daily_returns(symbol, start_date='2024-01-01', plot=True):

    query = """
            SELECT f.SYMBOL, f.DATE, f.OPEN, f.HIGH, f.LOW, f.CLOSE, f.VOLUME
            FROM NSEDATA_FACT f
            INNER JOIN METADATA m ON f.SYMBOL = m.SYMBOL 
            WHERE m.LISTING_DATE <= CURRENT_DATE - INTERVAL '65 days'
            AND f.DATE BETWEEN %(start_date)s AND %(end_date)s
            AND symbol = %(symbol)s;
        """

    df = pd.read_sql(query, engine, params={"start_date": start_date, "symbol": symbol})

    df['date'] = pd.to_datetime(df['date'])
    df = df.sort_values(by=['symbol', 'date'], ascending=[True, True])

    df['next_open'] = df['open'].shift(-1)
    df['next_date'] = df['date'].shift(-1)
    df['prev_date'] = df['date'].shift(1)
    df['prev_close'] = df['close'].shift(1)

    df['daily_return'] = df['close'] - df['prev_close']
    df['daily_return%'] = (df['daily_return']/df['prev_close'])*100

    mean_daily_return = df['daily_return'].dropna().mean()  
    annual_return = mean_daily_return*250  

    std_dev_daily_return = df['daily_return'].dropna().std()
    annual_std_dev = std_dev_daily_return * np.sqrt(250)

    buy_and_hold_return=(df[df['date'] == df['date'].max()]['close'].values[0]) - (df[df['date'] == start_date]['close'].values[0])

    if plot:
        
        plt.figure(figsize=(14, 6))

        # Daily Returns over time
        plt.subplot(1, 2, 1)
        plt.plot(df['date'], df['daily_return%'], label='Daily Return %', color='blue')
        plt.axhline(mean_daily_return, color='green', linestyle='--', label=f'Mean Daily Return ({mean_daily_return:.4f})')
        plt.title('Daily Returns Over Time')
        plt.xlabel('Date')
        plt.ylabel('Daily Return (%)')
        plt.legend()
        plt.grid(True)

        # Risk vs Return (Standard Deviation vs Mean Return)
        plt.subplot(1, 2, 2)
        plt.scatter(std_dev_daily_return, mean_daily_return, color='red')
        plt.text(std_dev_daily_return, mean_daily_return, f"  μ={mean_daily_return:.4f}, σ={std_dev_daily_return:.4f}", fontsize=12)
        plt.title('Risk vs Return')
        plt.xlabel('Risk (Std Dev)')
        plt.ylabel('Return (Mean)')
        plt.grid(True)

        plt.tight_layout()
        plt.show()

    return mean_daily_return, annual_return, std_dev_daily_return, annual_std_dev, buy_and_hold_return
