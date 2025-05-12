import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

import pandas as pd
import numpy as np
from sqlalchemy import create_engine
from utils import *

from config import SQL_ALCHEMY_CONN

engine = create_engine(SQL_ALCHEMY_CONN)

def dynamic_allocation(df, strategy, initial_capital,capital_exposure,buffer_pct, max_risk, commission=0.0005):
    
    dates_df = sorted(df['date'].unique())
    balance = initial_capital
    max_risk_per_trade = max_risk*balance
    buffer = buffer_pct*initial_capital
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

            if pd.notna(max_risk_per_trade) and stop_diff > 0:
                risk_adj_quantity = int(max_risk_per_trade//stop_diff)
            else:
                continue
            
            if pd.notna(score_calc_capital_for_stock) and pd.notna(row['price_adj']) and row['price_adj'] > 0:
                max_affordable_quantity = int(score_calc_capital_for_stock // row['price_adj'])
            else:
                continue
            quantity = min(max_affordable_quantity,risk_adj_quantity)
            commission_cost = quantity*price_adj*commission
            total_cost =  (quantity * price_adj) + commission_cost
        
            if row['order_type'] == 'Buy':
                if balance >= total_cost and quantity > 0 and balance > buffer:
                    balance-=total_cost
                    holdings[symbol] = holdings.get(symbol, 0) + quantity
                    trade_log.append({
                        "symbol": symbol,
                        "date": date,
                        "strategy": strategy,
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
                        "strategy": strategy,
                        "price_adj": price_adj,
                        "order_type": 'Sell',
                        "quantity": quantity,
                        "commision_cost": commission_cost,
                        "total_cost": sell_value,
                        "balance":balance
                    })
    return pd.DataFrame(trade_log)

def trade_log_insertion(trade_log):

    trade_log.to_sql(
        name='order_log',      # PostgreSQL table name
        con=engine,
        if_exists='append',          # Append rows to existing table
        index=False,                 # Do not insert DataFrame index
        method='multi'               # Efficient batch insert
)