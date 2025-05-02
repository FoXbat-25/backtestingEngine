import pandas as pd
import numpy as np
import matplotlib as plt
from config import SQL_ALCHEMY_CONN
from sqlalchemy import create_engine

from  datetime import datetime

from trade_book_copy import trade_book

engine = create_engine(SQL_ALCHEMY_CONN)

def fetch_nse_data(symbol, start_date='2024-01-01', end_date = datetime.today().date()):
    
        query = """
            SELECT f.SYMBOL, f.DATE, f.OPEN, f.HIGH, f.LOW, f.CLOSE, f.VOLUME
            FROM NSEDATA_FACT f
            INNER JOIN METADATA m ON f.SYMBOL = m.SYMBOL 
            WHERE m.LISTING_DATE <= CURRENT_DATE - INTERVAL '65 days'
            AND f.DATE BETWEEN %(start_date)s AND %(end_date)s
            AND symbol = %(symbol)s;
        """

        dataframe = pd.read_sql(query, engine, params={"start_date": start_date, "end_date": end_date, "symbol": symbol})

        dataframe['date'] = pd.to_datetime(dataframe['date'])
        dataframe = dataframe.sort_values(by=['symbol', 'date'], ascending=[True, True])

        dataframe['next_open'] = dataframe['open'].shift(-1)
        dataframe['next_date'] = dataframe['date'].shift(-1)
        dataframe['prev_date'] = dataframe['date'].shift(1)
        dataframe['prev_close'] = dataframe['close'].shift(1)

        return dataframe

def daily_returns(df, plot=True):

    df['daily_return'] = df['close'] - df['prev_close']
    df['daily_return%'] = (df['daily_return']/df['prev_close'])*100

    mean_daily_return = df['daily_return'].dropna().mean()  
    annual_return = mean_daily_return*250  

    std_dev_daily_return = df['daily_return'].dropna().std()
    annual_std_dev = std_dev_daily_return * np.sqrt(250)

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

    return mean_daily_return, annual_return, std_dev_daily_return, annual_std_dev
