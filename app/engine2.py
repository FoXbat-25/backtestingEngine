import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

import pandas as pd
import numpy as np
from sqlalchemy import create_engine
from sklearn.preprocessing import MinMaxScaler

from datetime import datetime, timedelta

from config import SQL_ALCHEMY_CONN

engine = create_engine(SQL_ALCHEMY_CONN)

def get_data(start_date='2023-01-01', end_date = datetime.today().date()):

    query = """
            SELECT f.SYMBOL, f.DATE, f.OPEN, f.HIGH, f.LOW, f.CLOSE, f.VOLUME
            FROM NSEDATA_FACT f
            INNER JOIN METADATA m ON f.SYMBOL = m.SYMBOL 
            WHERE m.LISTING_DATE <= CURRENT_DATE - INTERVAL '65 days'
            AND f.DATE BETWEEN %(start_date)s AND %(end_date)s;
        """

    df = pd.read_sql(query, engine, params={"start_date": start_date, "end_date": end_date})
    if df.empty:
            return {"error": "No closed trades found for this symbol/strategy"}

    df['date'] = pd.to_datetime(df['date'])
    df = df.sort_values(by=['symbol', 'date'], ascending=[True, True])

    df['prev_date'] = df.groupby('symbol')['date'].shift(1)
    df['prev_close'] = df.groupby('symbol')['close'].shift(1)

    df['daily_return'] = df['close'] - df['prev_close']
    df['daily_log_return'] = np.log(df['close']/df['prev_close'])
    df['daily_return%'] = (df['daily_return']/df['prev_close'])

    df['cumulative_log_return'] = df.groupby('symbol')['daily_log_return'].cumsum()
    df['cumulative_return'] = np.exp(df['cumulative_log_return'])

    df['cumulative_max'] = df.groupby('symbol')['cumulative_return'].cummax()
    df['drawdown'] = df['cumulative_return']/df['cumulative_max'] - 1

    return df

def get_daily_metrics(risk_free_rate):

    df = get_data()  
    df['date'] = pd.to_datetime(df['date'])
    
    grouped = df.groupby('symbol')

    # Basic return stats
    log_mean_daily_return = grouped['daily_log_return'].mean()
    std_dev_daily_return = grouped['daily_log_return'].std()
    annual_return = log_mean_daily_return * 250
    annual_std_dev = std_dev_daily_return * np.sqrt(250)

    # Sharpe
    sharpe_ratio = (annual_return - risk_free_rate) / annual_std_dev
    sharpe_ratio = sharpe_ratio.replace([np.inf, -np.inf], np.nan)

    # Max drawdown
    max_drawdown = grouped['drawdown'].min()

    # Buy-and-hold returns
    first_close = grouped.first()['close']
    last_close = grouped.last()['close']
    buy_and_hold_return = last_close - first_close
    buy_and_hold_return_pct = ((last_close / first_close) - 1) * 100

    # Win rate
    wins = df[df['daily_return'] > 0].groupby('symbol')['daily_return'].count()
    total = grouped['daily_return'].count()
    win_rate = wins / total

    # Combine into summary
    summary_df = pd.DataFrame({
        'annual_return': annual_return,
        'annual_volatility': annual_std_dev,
        'sharpe_ratio': sharpe_ratio,
        'max_drawdown': max_drawdown,
        'first_close': first_close,
        'last_close': last_close,
        'buy_and_hold_return': buy_and_hold_return,
        'buy_and_hold_return_pct': buy_and_hold_return_pct,
        'win_rate': win_rate
    })

    summary_df = summary_df.reset_index()

    scaler = MinMaxScaler()

    summary_df['sharpe_norm'] = scaler.fit_transform(summary_df[['sharpe_ratio']])
    summary_df['winrate_norm'] = scaler.fit_transform(summary_df[['win_rate']])
    summary_df['drawdown_norm'] = 1 - scaler.fit_transform(summary_df[['max_drawdown']])

    summary_df['score'] =  (0.5 * summary_df['sharpe_norm']) + (0.3 * summary_df['winrate_norm']) + (0.2 * (1 - summary_df['drawdown_norm']))
    return summary_df

def slippage(df, slippage_rate):

    df['entry_price_adj'] = df['entry_price'] * (1 + slippage_rate)
    df['exit_price_adj'] = df['exit_price'] * (1 - slippage_rate)

    return df

def get_indv_trades(df):
    
    buy_df = df[['symbol', 'entry_date', 'entry_price', 'strategy', 'created_at', 'updated_at', 'stop_loss', 'entry_price_adj', 'score']].copy()
    
    buy_df['price'] = buy_df['entry_price']
    buy_df['order_type'] = 'Buy'
    buy_df['date'] = pd.to_datetime(buy_df['entry_date'])
    buy_df['price_adj'] = buy_df['entry_price_adj']
    buy_df.drop(columns=['entry_date', 'entry_price', 'entry_price_adj'], inplace=True)

    sell_df = df[['symbol', 'exit_date', 'exit_price', 'strategy', 'created_at', 'updated_at', 'stop_loss', 'exit_price_adj', 'score']].copy()
    
    sell_df['price'] = sell_df['exit_price']
    sell_df['order_type'] = 'Sell'
    sell_df['date'] = pd.to_datetime(sell_df['exit_date'])
    sell_df['price_adj'] = sell_df['exit_price_adj']
    sell_df.drop(columns=['exit_date', 'exit_price', 'exit_price_adj'], inplace=True)

    event_df = pd.concat([buy_df, sell_df], ignore_index=True)

    return event_df
