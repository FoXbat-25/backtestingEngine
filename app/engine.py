import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

import pandas as pd
import numpy as np
from sqlalchemy import create_engine

from sklearn.preprocessing import MinMaxScaler

from datetime import datetime
import matplotlib as plt
from trade_book import trade_book

from config import SQL_ALCHEMY_CONN

engine = create_engine(SQL_ALCHEMY_CONN)

class backTester:

    def __init__(self, nATR, strategy_func, strategy_name):
        self.strategy_func = strategy_func      
        self.strategy_name = strategy_name
        self.nATR = nATR      
        self.populate_order_book()


    def populate_order_book(self):
        df = self.strategy_func()
        trade_book(self.nATR, df,strategy=self.strategy_name)

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
    df['daily_return%'] = (df['daily_return']/df['prev_close'])*100

    return df

def daily_returns(df, plot = False):

    start_date = df['date'].min()

    mean_daily_return = df['daily_return'].dropna().mean()  
    annual_return = mean_daily_return*250  

    std_dev_daily_return = df['daily_return'].dropna().std()
    annual_std_dev = std_dev_daily_return * np.sqrt(250)

    buy_and_hold_return= (df['close'].iloc[-1]) - (df[df['date'] == start_date]['close'].iloc[0])
    buy_and_hold_return_pct = (buy_and_hold_return/(df[df['date'] == start_date]['close'].iloc[0]))*100

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

    metric =  {
        "mean daily return": mean_daily_return,
        "annual return": annual_return,
        "std_dev_daily_return": std_dev_daily_return,
        "annual_std_dev": annual_std_dev,
        "buy_and_hold_return": buy_and_hold_return,
        "buy_and_hold_return_pct": buy_and_hold_return_pct
    }

    metrics_df = pd.DataFrame([metric]) 
    return metrics_df


def order_book_transformation(symbol, strategy_name,initial_cap, slippage_rate = 0.001,max_risk = 0.01, commission = 0.0005, start_date = '2023-01-01'):
        
        query="""
            SELECT * 
            FROM TRADE_BOOK 
            WHERE symbol = %(symbol)s and strategy = %(strategy_name)s 
            and entry_date >= %(start_date)s and status = 'closed';
        """        

        trade_df = pd.read_sql(query, engine, params={"symbol": symbol, "strategy_name": strategy_name, "start_date":start_date}) 
        
        # max_cap_per_trade = max_risk*initial_cap

        trade_df['entry_date'] = pd.to_datetime(trade_df['entry_date'])
        trade_df = trade_df.sort_values(by='entry_date', ascending= True)

        trade_df['symbol'] = symbol

        trade_df['entry_price_adj'] = trade_df['entry_price'] * (1 + slippage_rate)
        trade_df['exit_price_adj'] = trade_df['exit_price'] * (1 - slippage_rate)

        trade_df['stop_diff'] = trade_df['entry_price_adj'] - trade_df['stop_loss']

        trade_df = trade_df[(trade_df['stop_diff'] > 0) & (trade_df['entry_price_adj'] > 0)]  # Ensure valid stop losses

        trade_df['pl_adj'] = (trade_df['exit_price_adj'] - trade_df['entry_price_adj']) # * trade_df['quantity']

        trade_df['daily_return'] = trade_df['pl_adj']/trade_df['holding_period']
        
        trade_df['daily_return%'] = (trade_df['daily_return']/trade_df['entry_price'])

        trade_df['cumulative_return'] = (1 + trade_df['daily_return%']).cumprod()
        trade_df['cumulative_max'] = trade_df['cumulative_return'].cummax()
        trade_df['drawdown'] = trade_df['cumulative_return'] / trade_df['cumulative_max'] - 1

        # trade_df['quantity'] = (max_cap_per_trade // trade_df['stop_diff']).astype(int)
        # trade_df['max_affordable_qty'] = (initial_cap // trade_df['entry_price_adj']).astype(int)
        # trade_df['quantity'] = trade_df[['quantity', 'max_affordable_qty']].min(axis=1)

        # trade_df = trade_df[trade_df['quantity'] > 0]
        # trade_df['commission_cost'] = trade_df['quantity'] * (trade_df['entry_price_adj'] + trade_df['exit_price_adj']) * commission
        # trade_df['net_pl'] = trade_df['pl_adj'] - trade_df['commission_cost']
        # trade_df['order_cost'] = (trade_df['entry_price_adj'] * trade_df['quantity']) + trade_df['commission_cost']       

        return trade_df

def order_book_metrics(trade_df,initial_cap, risk_free_rate = 0.07, confidence_level = 0.95):

    max_drawdown = trade_df['drawdown'].min()
    mean_daily_return = trade_df['daily_return'].dropna().mean()  
    annual_return = mean_daily_return*250

    std_dev_daily_return = trade_df['daily_return'].dropna().std()
    annual_volatility = std_dev_daily_return * np.sqrt(250)

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

        trade_df = order_book_transformation(symbol, strategy, initial_capital)
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

def normalisation(df):

    scaler = MinMaxScaler()

    df['sharpe_norm'] = scaler.fit_transform(df[['sharpe_ratio']])
    df['winrate_norm'] = scaler.fit_transform(df[['win_rate']])
    df['drawdown_norm'] = 1 - scaler.fit_transform(df[['max_drawdown']])

    df['score'] =  (0.5 * df['sharpe_norm']) + (0.3 * df['winrate_norm']) + (0.2 * (1 - df['drawdown_norm']))
    
    df['symbol'] = df['symbol'].str[0]
    # all_orders_df=all_orders_df.merge(df[['symbol','score']], on='symbol', how='left')

    return df

def dynamic_allocation(df, initial_capital,capital_exposure, max_risk, commission=0.0005):
    
    dates_df = df['date'].unique()
    balance = initial_capital
    max_risk_per_trade = max_risk*balance
    trade_log=[]
    holdings={}
    
    for date in dates_df:
        
        data = df[df['date']==date].sort_values('score', ascending = False)
        capital_for_day = capital_exposure*balance
        data['score_norm'] = data['score']/data['score'].sum()
        
        for _, row in data.iterrows():
            
            symbol=row['symbol']
            price_adj = row['price_adj']
            score_calc_capital_for_stock=row['score_norm']*capital_for_day
            stop_diff = max(price_adj - row['stop_loss'], 0.02)
            risk_adj_quantity = int(max_risk_per_trade//stop_diff)
            max_affordable_quantity = int(score_calc_capital_for_stock//row['price_adj'])
            quantity = min(max_affordable_quantity,risk_adj_quantity)
            commission_cost = quantity*price_adj*commission
            total_cost =  (quantity * price_adj) + commission_cost
        
            if row['order_type'] == 'Buy':
                if balance >= total_cost:
                    balance-=total_cost
                    holdings[symbol] = holdings.get(symbol, 0) + quantity
                    trade_log.append({
                        "symbol": symbol,
                        "date": date,
                        "price_adj": price_adj,
                        "order_type": 'Buy',
                        "quantity": quantity,
                        "commision_cost": commission_cost,
                        "total_cost":total_cost,
                        "balance":balance
                    })
            elif row['order_type'] == 'Sell':
                held_qty = holdings.get(symbol, 0)
                if held_qty > 0:
                    quantity = held_qty
                    holdings[symbol] = 0
                    sell_value = (quantity*price_adj) - commission_cost
                    balance += sell_value 
                    trade_log.append({
                        "symbol": symbol,
                        "date": date,
                        "price_adj": price_adj,
                        "order_type": 'Sell',
                        "quantity": quantity,
                        "commision_cost": commission_cost,
                        "total_cost": sell_value,
                        "balance":balance
                    })
    return pd.DataFrame(trade_log)
            
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
    

# def dynamic_allocation(final_df):


# def position_sizing(trade_df, initial_cap, max_risk = 0.01, commission = 0.0005):
    
#     max_cap_per_trade = max_risk*initial_cap

#     trade_df['stop_diff'] = trade_df['entry_price_adj'] - trade_df['stop_loss']
#     trade_df = trade_df[(trade_df['stop_diff'] > 0) & (trade_df['entry_price_adj'] > 0)]  # Ensure valid stop losses

#     trade_df['quantity'] = (max_cap_per_trade // trade_df['stop_diff']).astype(int)
#     trade_df['max_affordable_qty'] = (initial_cap // trade_df['entry_price_adj']).astype(int)
#     trade_df['quantity'] = trade_df[['quantity', 'max_affordable_qty']].min(axis=1)
#     trade_df['pl_adj'] = (trade_df['exit_price_adj'] - trade_df['entry_price_adj']) * trade_df['quantity']

#     trade_df['daily_return'] = trade_df['pl_adj']/trade_df['holding_period']
        
#     trade_df['daily_return%'] = (trade_df['daily_return']/trade_df['entry_price'])

#     trade_df['cumulative_return'] = (1 + trade_df['daily_return%']).cumprod()
#     trade_df['cumulative_max'] = trade_df['cumulative_return'].cummax()
#     trade_df['drawdown'] = trade_df['cumulative_return'] / trade_df['cumulative_max'] - 1
    
#     trade_df = trade_df[trade_df['quantity'] > 0]
#     trade_df['commission_cost'] = trade_df['quantity'] * (trade_df['entry_price_adj'] + trade_df['exit_price_adj']) * commission
#     trade_df['net_pl'] = trade_df['pl_adj'] - trade_df['commission_cost']
#     trade_df['order_cost'] = (trade_df['entry_price_adj'] * trade_df['quantity']) + trade_df['commission_cost']

#     for _, row in trade_df.iterrows():
        
#         balance = initial_cap
#         balances = []

#         if row['order_cost'] <= balance:
#             balance -= row['order_cost']
        
#         else: 
#             print("No money for trades")
        
#         balances.append(balance)

#     trade_df['balance'] = balances

#     return trade_df


# balance = initial_cap
#         balances = []
#         invested = []
#         executed = []
# for _, row in trade_df.iterrows():
#             if row['order_cost'] <= balance:
#                 balance -= row['order_cost']
#                 balances.append(balance)
#                 executed.append(True)
#                 invested.append(row['order_cost'])
#             else:
#                 balances.append(balance)  # or np.nan if you prefer
#                 executed.append(False)
#                 invested.append(0)

#         trade_df['balances'] = balances
#         trade_df['executed'] = executed
#         trade_df['invested'] = invested