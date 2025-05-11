import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

import pandas as pd
import numpy as np
from sqlalchemy import create_engine

from utils import *

from config import SQL_ALCHEMY_CONN

engine = create_engine(SQL_ALCHEMY_CONN)

def order_book_transformation(symbol, strategy_name, slippage_rate = 0.001, start_date = '2023-01-01'):
        
        query="""
            SELECT * 
            FROM TRADE_BOOK 
            WHERE symbol = %(symbol)s and strategy = %(strategy_name)s 
            and entry_date >= %(start_date)s and status = 'closed';
        """        

        trade_df = pd.read_sql(query, engine, params={"symbol": symbol, "strategy_name": strategy_name, "start_date":start_date}) 
        
        trade_df['entry_date'] = pd.to_datetime(trade_df['entry_date'])
        trade_df = trade_df.sort_values(by='entry_date', ascending= True)

        trade_df['symbol'] = symbol

        trade_df['entry_price_adj'] = trade_df['entry_price'] * (1 + slippage_rate)
        trade_df['exit_price_adj'] = trade_df['exit_price'] * (1 - slippage_rate)

        trade_df['stop_diff'] = trade_df['entry_price_adj'] - trade_df['stop_loss']

        trade_df = trade_df[(trade_df['stop_diff'] > 0) & (trade_df['entry_price_adj'] > 0)]  # Ensure valid stop losses

        trade_df['log_return'] = np.log(trade_df['exit_price_adj']/trade_df['entry_price_adj'])
        trade_df['daily_log_return'] = trade_df['log_return']/trade_df['holding_period']

        trade_df['daily_return%'] = np.exp(trade_df['daily_log_return']) - 1

        trade_df['cumulative_log_return'] = trade_df['daily_log_return'].cumsum()
        trade_df['cumulative_return'] = np.exp(trade_df['cumulative_log_return'])
        
        trade_df['cumulative_max'] = trade_df['cumulative_return'].cummax()
        trade_df['drawdown'] = trade_df['cumulative_return'] / trade_df['cumulative_max'] - 1

        trade_df['pl_adj'] = (trade_df['exit_price_adj'] - trade_df['entry_price_adj'])     

        return trade_df

def daily_metrics(df, risk_free_rate):

    start_date = df['date'].min()
    symbol = df['symbol'].unique()[0]
    log_mean_daily_return = df['daily_log_return'].dropna().mean()  
    annual_return = log_mean_daily_return*250  
    df['cumulative_log_return'] = df['daily_log_return'].cumsum()
    df['cumulative_return'] = np.exp(df['cumulative_log_return'])
    df['cumulative_max'] = df['cumulative_return'].cummax()
    df['drawdown'] = df['cumulative_return']/df['cumulative_max'] - 1
    max_drawdown = df['drawdown'].min()

    std_dev_daily_return = df['daily_log_return'].dropna().std()
    annual_std_dev = std_dev_daily_return * np.sqrt(250)

    if annual_std_dev == 0 or np.isnan(annual_std_dev):
        sharpe_ratio = np.nan
    else:
        sharpe_ratio = (annual_return - risk_free_rate)/annual_std_dev

    total_trades = len(df)
    win_rate = (len(df[df['daily_return'] > 0]))/total_trades if total_trades else np.nan

    buy_and_hold_return= (df['close'].iloc[-1]) - (df[df['date'] == start_date]['close'].iloc[0])
    buy_and_hold_return_pct = (buy_and_hold_return/(df[df['date'] == start_date]['close'].iloc[0]))*100

    metric =  {
        "symbol": symbol,
        "log mean daily return": log_mean_daily_return,
        "log annual return": annual_return,
        "log std_dev_daily_return": std_dev_daily_return,
        "log annual_std_dev": annual_std_dev,
        "buy_and_hold_return": buy_and_hold_return,
        "buy_and_hold_return_pct": buy_and_hold_return_pct,
        "sharpe_ratio": sharpe_ratio,
        "max_drawdown": max_drawdown,
        "win_rate" : win_rate
    }

    metrics_df = pd.DataFrame([metric]) 
    return metrics_df

def all_daily_metrics(risk_free_rate):
    
    results_list = []
    
    symbols_df = fetch_symbols()
    symbols_df = symbols_df['symbol'].unique()
    for symbol in symbols_df: 
        data_df = fetch_data(symbol)
        if not data_df.empty:
            results = daily_metrics(data_df, risk_free_rate)
            results_list.append(results)

        else:
            print(f'{symbol} : error - no trades')
        
    final_df = pd.concat(results_list, ignore_index=True)
    final_df = normalisation(final_df)
    return final_df

def order_book_metrics(trade_df,initial_cap, risk_free_rate = 0.07, confidence_level = 0.95):

    max_drawdown = trade_df['drawdown'].min()
    log_mean_daily_return = trade_df['daily_log_return'].dropna().mean()  
    annual_return = log_mean_daily_return*250

    log_std_dev_daily_return = trade_df['daily_log_return'].dropna().std()
    annual_volatility = log_std_dev_daily_return * np.sqrt(250)

    if annual_volatility == 0 or np.isnan(annual_volatility):
        sharpe_ratio = np.nan
    else:
        sharpe_ratio = (annual_return - risk_free_rate)/annual_volatility

    wins = trade_df[trade_df['pl_adj'] > 0]
    losses = trade_df[trade_df['pl_adj'] < 0]
    total_pl = trade_df['pl_adj'].sum()
    # total_investment = trade_df['order_cost'].sum()
    # return_pct = total_pl/total_investment
    total_days_held = trade_df['holding_period'].sum() 
    
    symbol = trade_df['symbol'].unique().tolist()
    num_wins = len(wins)
    num_losses = len(losses)
    total_trades = len(trade_df)
    first_trade = trade_df['entry_date'].min()
    win_rate = len(wins) / total_trades if total_trades else np.nan
    avg_win = wins['pl_adj'].mean() if not wins.empty else 0
    avg_loss = losses['pl_adj'].mean() if not losses.empty else 0
    profit_factor = wins['pl_adj'].sum() / abs(losses['pl_adj'].sum()) if not losses.empty else np.inf
    payoff_ratio = avg_win / abs(avg_loss) if avg_loss != 0 else np.inf
    expectancy = (win_rate * avg_win) + ((1 - win_rate) * avg_loss)

    VaR_1d = np.percentile(trade_df['daily_return%'], (1 - confidence_level) * 100)
    VaR_1d_value = initial_cap * abs(VaR_1d) 

    if total_days_held == 0:
        cagr = np.nan  
    else:
        final_value = trade_df['cumulative_return'].iloc[-1]
        years_held = total_days_held / 365

        if pd.notna(final_value) and final_value > 0 and years_held > 0:
            cagr = final_value ** (1 / years_held) - 1
        else:
            cagr = np.nan

    metric =  {
                "symbol": symbol,
                "total_pl": total_pl,
                "total_days_held": total_days_held,
                "annual_return": annual_return,
                "annual_volatility": annual_volatility,
                "sharpe_ratio": sharpe_ratio,
                "total_wins": num_wins,
                "total_losses": num_losses,
                "cumulative_return": trade_df['cumulative_return'].iloc[-1],
                "cagr":cagr,
                "max_drawdown": max_drawdown,
                "win_rate": win_rate,
                "average_win": avg_win,
                "average_loss": avg_loss,
                "profit_factor": profit_factor,
                "payoff_ratio": payoff_ratio,
                "expectancy": expectancy,
                "total_trades": total_trades,
                "first_trade": first_trade,
                "last_trade": trade_df['entry_date'].max(),
                "VaR": VaR_1d_value
            }

    metrics_df = pd.DataFrame([metric])    
    return metrics_df


def all_orders_and_metrics(strategy, initial_capital):
    
    results_list = []
    orders_list = []

    symbols_df = fetch_symbols()
    symbols_df = symbols_df['symbol'].unique()
    for symbol in symbols_df:   

        trade_df = order_book_transformation(symbol, strategy)
        if not trade_df.empty:
            results = order_book_metrics(trade_df, initial_capital)
            results_list.append(results)
            orders_list.append(indv_trade_listing(trade_df))
        
        else:
            print(f'No trade for symbol: {symbol}.')
        
    final_df = pd.concat(results_list, ignore_index=True)
    final_df = normalisation(final_df)
    all_orders_df = pd.concat(orders_list, ignore_index=True)
    
    return final_df, all_orders_df

def all_orders(strategy, slippage_rate , start_date):

    query = """
        SELECT * 
            FROM TRADE_BOOK 
            WHERE strategy = %(strategy_name)s 
            and entry_date >= %(start_date)s and status = 'closed';
    """
    df = pd.read_sql(query, engine, params={ "strategy_name": strategy, "start_date":start_date}) 
    df['entry_date'] = pd.to_datetime(df['entry_date'])
    df = df.sort_values(by='entry_date', ascending= True)

    df['entry_price_adj'] = df['entry_price'] * (1 + slippage_rate)
    df['exit_price_adj'] = df['exit_price'] * (1 - slippage_rate)



    return df

def daily_returns(start_date, end_date, risk_free_rate):

    query = """
            SELECT f.SYMBOL, f.DATE, f.OPEN, f.HIGH, f.LOW, f.CLOSE, f.VOLUME
            FROM NSEDATA_FACT f
            INNER JOIN METADATA m ON f.SYMBOL = m.SYMBOL 
            WHERE m.LISTING_DATE <= CURRENT_DATE - INTERVAL '65 days'
            AND f.DATE BETWEEN %(start_date)s AND %(end_date)s;
        """

    dataframe = pd.read_sql(query, engine, params={"start_date": start_date, "end_date": end_date})

    dataframe['date'] = pd.to_datetime(dataframe['date'])
    dataframe = dataframe.sort_values(by=['symbol', 'date'], ascending=[True, True])

    dataframe['prev_close'] = dataframe.groupby('symbol')['close'].shift(1)
    dataframe['next_date'] = dataframe.groupby('symbol')['date'].shift(-1)
    dataframe['prev_date'] = dataframe.groupby('symbol')['date'].shift(1)

    dataframe['daily_return'] = dataframe['close'] - dataframe['prev_close']
    dataframe['daily_log_return'] = np.log(dataframe['close']/dataframe['prev_close'])
    dataframe['daily_return%'] = (dataframe['daily_return']/dataframe['prev_close'])

    log_mean_daily_return = dataframe.groupby('symbol')['daily_log_return'].dropna().mean()
    annual_return = log_mean_daily_return*250  
    dataframe['cumulative_log_return'] = dataframe.groupby('symbol')['daily_log_return'].cumsum()
    dataframe['cumulative_return'] = np.exp(['cumulative_log_return'])
    dataframe['cumulative_max'] = dataframe.groupby('symbol')['cumulative_return'].cummax()
    dataframe['drawdown'] = dataframe['cumulative_return']/dataframe['cumulative_max'] - 1
    max_drawdown = dataframe.groupby('symbol')['drawdown'].min()

    std_dev_daily_return = dataframe.groupby('symbol')['daily_log_return'].dropna().std()
    annual_std_dev = std_dev_daily_return * np.sqrt(250)

    if annual_std_dev == 0 or np.isnan(annual_std_dev):
        sharpe_ratio = np.nan
    else:
        sharpe_ratio = (annual_return - risk_free_rate)/annual_std_dev

    total_trades = len(dataframe)
    win_rate = (len(dataframe[dataframe.groupby('symbol')['daily_return'] > 0]))/total_trades if total_trades else np.nan

    buy_and_hold_return= (dataframe.groupby('symbol')['close'].iloc[-1]) - (dataframe[dataframe.groupby('symbol')['date'] == start_date]['close'].iloc[0])
    buy_and_hold_return_pct = (buy_and_hold_return/(dataframe[dataframe.groupby('symbol')['date'] == start_date]['close'].iloc[0]))*100

    return dataframe

def indexes_metrics(first_trade_date, last_trade_date, symbol = "^NSEI"):

    indexes_query= """
            SELECT * FROM INDEXES_FACT WHERE SYMBOL = %(symbol)s and date BETWEEN %(start_date)s and %(end_date)s; 
    """
    indexes_df = pd.read_sql(indexes_query, engine, params={"symbol":symbol, "start_date":first_trade_date, "end_date": last_trade_date})
    indexes_df['date'] = pd.to_datetime(indexes_df['date'])
    indexes_df = indexes_df.sort_values(by='date', ascending= True)
    
    latest_date = indexes_df['date'].max()
    start_price = indexes_df[indexes_df['date'] >= first_trade_date]['close'].iloc[0]
    end_price = indexes_df[indexes_df['date'] <= latest_date]['close'].iloc[-1]

    buy_and_hold_return = end_price - start_price
    buy_and_hold_return_pct = (end_price / start_price - 1) * 100

    return {
        "symbol": symbol,
        "start_price": start_price,
        "end_price": end_price,
        "buy_and_hold_return": buy_and_hold_return,
        "return_pct": buy_and_hold_return_pct
    }