import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

import pandas as pd
import numpy as np
from sqlalchemy import create_engine

from sklearn.preprocessing import MinMaxScaler

from datetime import datetime
from trade_book import trade_book

from config import SQL_ALCHEMY_CONN

engine = create_engine(SQL_ALCHEMY_CONN)

class backTester:

    def __init__(self, nATR, strategy_func, strategy_name, start_from='2023-01-01', adx_window=15,atr_window=14, z_score_window_list=[20], min_volume=500000, vol_window = 20, di_diff_window=20):
        self.strategy_func = strategy_func      
        self.strategy_name = strategy_name
        self.nATR = nATR
        self.start_date = start_from
        self.adx_window = adx_window
        self.atr_window = atr_window
        self.z_score_window_list = z_score_window_list
        self.min_volume =  min_volume
        self.vol_window = vol_window
        self.di_diff_window = di_diff_window     
        self.populate_order_book()


    def populate_order_book(self):
        self.df = self.strategy_func(self.start_date, self.adx_window, self.atr_window, self.z_score_window_list, self.min_volume, self.vol_window, self.di_diff_window)
        self.trades_df = trade_book(self.nATR, self.df,strategy=self.strategy_name)
        return self.trades_df
    
def fetch_symbols():

    query="""
        SELECT DISTINCT SYMBOL FROM TRADE_BOOK;
    """
    df = pd.read_sql(query, engine)
    df = df.sort_values(by=['symbol'], ascending=[True])
    return df

def fetch_data(symbol, start_date='2023-01-01', end_date = datetime.today().date()):

    query = """
            SELECT f.SYMBOL, f.DATE, f.OPEN, f.HIGH, f.LOW, f.CLOSE, f.VOLUME
            FROM NSEDATA_FACT f
            INNER JOIN METADATA m ON f.SYMBOL = m.SYMBOL 
            WHERE m.LISTING_DATE <= CURRENT_DATE - INTERVAL '65 days'
            AND f.DATE BETWEEN %(start_date)s AND %(end_date)s
            AND f.symbol = %(symbol)s;
        """

    df = pd.read_sql(query, engine, params={"start_date": start_date, "symbol": symbol, "end_date": end_date})
    if df.empty:
            return {"error": "No closed trades found for this symbol/strategy"}

    df['date'] = pd.to_datetime(df['date'])
    df = df.sort_values(by=['symbol', 'date'], ascending=[True, True])

    df['next_open'] = df['open'].shift(-1)
    df['next_date'] = df['date'].shift(-1)
    df['prev_date'] = df['date'].shift(1)
    df['prev_close'] = df['close'].shift(1)

    df['daily_return'] = df['close'] - df['prev_close']
    df['daily_log_return'] = np.log(df['close']/df['prev_close'])
    df['daily_return%'] = (df['daily_return']/df['prev_close'])*100

    return df

def normalisation(df):

    scaler = MinMaxScaler()

    df['sharpe_norm'] = scaler.fit_transform(df[['sharpe_ratio']])
    df['winrate_norm'] = scaler.fit_transform(df[['win_rate']])
    df['drawdown_norm'] = 1 - scaler.fit_transform(df[['max_drawdown']])

    df['score'] =  (0.5 * df['sharpe_norm']) + (0.3 * df['winrate_norm']) + (0.2 * (1 - df['drawdown_norm']))

    return df

def indv_trade_listing(df):

    buy_df = df[['symbol', 'entry_date', 'entry_price', 'strategy', 'created_at', 'updated_at', 'stop_loss', 'entry_price_adj']].copy()
    
    buy_df['price'] = buy_df['entry_price']
    buy_df['order_type'] = 'Buy'
    buy_df['date'] = pd.to_datetime(buy_df['entry_date'])
    buy_df['price_adj'] = buy_df['entry_price_adj']
    buy_df.drop(columns=['entry_date', 'entry_price', 'entry_price_adj'], inplace=True)

    sell_df = df[['symbol', 'exit_date', 'exit_price', 'strategy', 'created_at', 'updated_at', 'stop_loss', 'exit_price_adj']].copy()
    
    sell_df['price'] = sell_df['exit_price']
    sell_df['order_type'] = 'Sell'
    sell_df['date'] = pd.to_datetime(sell_df['exit_date'])
    sell_df['price_adj'] = sell_df['exit_price_adj']
    sell_df.drop(columns=['exit_date', 'exit_price', 'exit_price_adj'], inplace=True)

    event_df = pd.concat([buy_df, sell_df], ignore_index=True)
    # event_df = event_df.sort_values(by=['date', 'symbol'])
    # event_df = event_df.merge(all_orders_df[['symbol', 'score']], on='symbol', how='left')

    return event_df

def read_ff_factors():
    
    factors_df = pd.read_csv(
    '../data/F-F_Research_Data_Factors_daily.CSV', skiprows=3,skipfooter=10,engine='python', index_col=0
    )
    factors_df.index = pd.to_datetime(factors_df.index, format="%Y%m%d")
    factors_df = factors_df.rename(columns={
        'Mkt-RF': 'mkt_excess',
        'SMB': 'smb',
        'HML': 'hml',
        'RF': 'rf'
    }) / 100 

    return factors_df