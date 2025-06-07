import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

import pandas as pd
import numpy as np
from sqlalchemy import create_engine
from utils import *

import psycopg2
from psycopg2.extras import execute_values

from config import SQL_ALCHEMY_CONN, SQL_POSTGRES_CONN

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
            total_buy_cost =  (quantity * price_adj) + commission_cost
        
            if row['order_type'] == 'Buy':
                if balance >= total_buy_cost and quantity > 0 and balance > buffer:
                    balance-=total_buy_cost
                    holdings[symbol] = holdings.get(symbol, 0) + quantity
                    trade_log.append({
                        "symbol": symbol,
                        "date": date,
                        "strategy": strategy,
                        "price_adj": price_adj,
                        "order_type": 'Buy',
                        "quantity": quantity,
                        "commision_cost": commission_cost,
                        "total_cost":total_buy_cost,
                        "balance":balance
                    })
            elif row['order_type'] == 'Sell':
                held_qty = holdings.get(symbol, 0)
                if held_qty > 0:
                    quantity = held_qty
                    commission_cost = quantity*price_adj*commission
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

    conn = psycopg2.connect(SQL_POSTGRES_CONN)
    cursor = conn.cursor()
    
    orders = [
        (
            row['symbol'],
            row['date'].date() if isinstance(row['date'], pd.Timestamp) else row['date'],
            row['strategy'],
            float(row['price_adj']),
            row['order_type'],
            int(row['quantity']),
            float(row['commision_cost']),
            float(row['total_cost']),
            float(row['balance'])
        )
        for _, row in trade_log.iterrows()
    ]
    # Build the SQL INSERT query
    
    cursor.execute("TRUNCATE TABLE order_logs;")
    
    insert_query = """
    INSERT INTO order_logs (symbol, date, strategy, price_adj, order_type, quantity, commision_cost, total_cost, balance)
    VALUES %s
    ON CONFLICT (symbol, date)
    DO NOTHING
    """

    # Use execute_values for efficient batch insert
    execute_values(cursor, insert_query, orders)

    # Commit and close
    conn.commit()
    cursor.close()
    conn.close()

def get_portfolio_log(initial_capital):

    query = """
        WITH buys AS (
            SELECT
                symbol,
                date AS entry_date,
                price_adj AS entry_price,
                quantity,
                total_cost,
                commision_cost,
                ROW_NUMBER() OVER (PARTITION BY symbol ORDER BY date) AS rn
            FROM order_logs
            WHERE order_type = 'Buy'
        ),
        sells AS (
            SELECT
                symbol,
                date AS exit_date,
                price_adj AS exit_price,
                quantity,
                total_cost,
                commision_cost,
                ROW_NUMBER() OVER (PARTITION BY symbol ORDER BY date) AS rn
            FROM order_logs
            WHERE order_type = 'Sell'
        ),
        order_summary as (
        SELECT
            b.symbol,
            b.entry_date,
            b.entry_price,
            s.exit_date,
            s.exit_price,
            b.quantity,
            b.quantity * b.entry_price AS total_invested,
            b.quantity * s.exit_price AS total_sold,
            b.commision_cost AS buy_commission,
            s.commision_cost AS sell_commision,
            b.commision_cost + s.commision_cost AS total_commision
        FROM buys b
        JOIN sells s
        ON b.symbol = s.symbol AND b.rn = s.rn
        ORDER BY b.symbol, b.entry_date
        ),
        date_range AS (
            SELECT DISTINCT date
            FROM nsedata_fact WHERE date >= '2020-01-01'
        ),
        holding_days AS (
            SELECT
                d.date,
                h.symbol,
                h.quantity
            FROM order_summary h
            JOIN date_range d
            ON d.date BETWEEN h.entry_date AND h.exit_date
        ),
        valuation AS (
            SELECT
                hd.date,
                hd.symbol,
                hd.quantity,
                nf.close,
                hd.quantity * nf.close AS holding_value
            FROM holding_days hd
            JOIN nsedata_fact nf
            ON hd.symbol = nf.symbol AND hd.date = nf.date
        ),
        daily_valuation AS (
            SELECT
                date,
                SUM(holding_value) AS portfolio_value
            FROM valuation
            GROUP BY date
        ),
        daily_investment AS (
            SELECT
                entry_date AS date,
                --SUM(entry_price * quantity) AS total_investment,
                SUM(total_invested + buy_commission) as total_buy_cost
            FROM order_summary
            GROUP BY entry_date
        ),
        daily_sold AS (
            SELECT 
                exit_date AS date,
                SUM(exit_price * quantity) AS total_exit,
                SUM(total_sold - sell_commision) AS total_sell_value
            FROM order_summary
            GROUP BY exit_date
        )
        SELECT
            v.date,
            v.portfolio_value,
            --COALESCE(i.total_investment, 0) AS total_investment,
            COALESCE(i.total_buy_cost, 0) AS total_buy_cost,
            COALESCE(m.total_sell_value, 0) AS total_sell_value
        FROM daily_valuation v
        LEFT JOIN daily_investment i ON v.date = i.date
        LEFT JOIN daily_sold m ON v.date = m.date
        ORDER BY v.date;
    """

    df = pd.read_sql(query, engine)
    df['net_flow'] = df['total_sell_value'] - df['total_buy_cost']
    df['balance'] = initial_capital + df['net_flow'].cumsum()
    df['asset_valuation'] = df['balance'] + df['portfolio_value']
    return df