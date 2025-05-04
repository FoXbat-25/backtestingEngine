import pandas as pd
import numpy as np
from sqlalchemy import create_engine

from datetime import datetime
import matplotlib as plt
from meanReversion.app.mean_reversion import mean_reversion
from trade_book import trade_book

from config import SQL_ALCHEMY_CONN

engine = create_engine(SQL_ALCHEMY_CONN)

class backTester:

    def __init__(self, strategy_func, strategy_name):
        self.strategy_func = strategy_func      # Save the strategy function
        self.strategy_name = strategy_name      # Save the name for logging/trade book
        self.populate_order_book()


    def populate_order_book(self):
        df = self.strategy_func()
        trade_book(df,strategy=self.strategy_name)

def fetch_data(symbol, start_date='2024-01-01', end_date = datetime.today().date()):

    query = """
            SELECT f.SYMBOL, f.DATE, f.OPEN, f.HIGH, f.LOW, f.CLOSE, f.VOLUME
            FROM NSEDATA_FACT f
            INNER JOIN METADATA m ON f.SYMBOL = m.SYMBOL 
            WHERE m.LISTING_DATE <= CURRENT_DATE - INTERVAL '65 days'
            AND f.DATE BETWEEN %(start_date)s AND %(end_date)s
            AND symbol = %(symbol)s;
        """

    df = pd.read_sql(query, engine, params={"start_date": start_date, "symbol": symbol, "end_date": end_date})

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

    buy_and_hold_return=(df[df['date'] >= start_date]['close'].iloc[0]) - (df['close'].iloc[-1])

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

    return {
        "mean daily return": mean_daily_return,
        "annual return": annual_return,
        "std_dev_daily_return": std_dev_daily_return,
        "annual_std_dev": annual_std_dev,
        "buy_and_hold_return": buy_and_hold_return
    }

def indexes_return(first_trade_date, last_trade_date, symbol = "^NSEI"):

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

def metrics(trade_df,initial_cap, risk_free_rate = 0.07, confidence_level = 0.95):

    max_drawdown = trade_df['drawdown'].min()
    mean_daily_return = trade_df['daily_return%'].dropna().mean()  
    annual_return = mean_daily_return*250

    std_dev_daily_return = trade_df['daily_return%'].dropna().std()
    annual_volatility = std_dev_daily_return * np.sqrt(250)

    if annual_volatility == 0 or np.isnan(annual_volatility):
        sharpe_ratio = np.nan
    else:
        sharpe_ratio = (annual_return - risk_free_rate)/annual_volatility

    wins = trade_df[trade_df['pl_adj'] > 0]
    losses = trade_df[trade_df['pl_adj'] < 0]
    total_pl = trade_df['pl_adj'].sum()
    total_days_held = trade_df['holding_period'].sum() 
    
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
        cagr = np.nan  # Avoid division by zero
    else:
        final_value = trade_df['cumulative_return'].iloc[-1]
        years_held = total_days_held / 365
        cagr = final_value ** (1 / years_held) - 1 if years_held > 0 else np.nan

    return {

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

def order_book_transformation(symbol, strategy_name, slippage_rate = 0.001, start_date = '2024-01-01'):
        
        query="""
            SELECT * 
            FROM TRADE_BOOK 
            WHERE symbol = %(symbol)s and strategy = %(strategy_name)s 
            and entry_date >= %(start_date)s and status = 'closed';
        """        

        trade_df = pd.read_sql(query, engine, params={"symbol": symbol, "strategy": strategy_name, "start_date":start_date})

        if trade_df.empty:
            return {"error": "No closed trades found for this symbol/strategy"}

        trade_df['entry_date'] = pd.to_datetime(trade_df['entry_date'])
        trade_df = trade_df.sort_values(by='entry_date', ascending= True)

        trade_df['entry_price_adj'] = trade_df['entry_price'] * (1 + slippage_rate)
        trade_df['exit_price_adj'] = trade_df['exit_price'] * (1 - slippage_rate)
        trade_df['pl_adj'] = (trade_df['exit_price_adj'] - trade_df['entry_price_adj']) * trade_df['qty']


        trade_df['daily_return'] = trade_df['pl_adj']/trade_df['holding_period']
        
        trade_df['daily_return%'] = (trade_df['daily_return']/trade_df['entry_price'])

        trade_df['cumulative_return'] = (1 + trade_df['daily_return%']).cumprod()
        trade_df['cumulative_max'] = trade_df['cumulative_return'].cummax()
        trade_df['drawdown'] = trade_df['cumulative_return'] / trade_df['cumulative_max'] - 1
        
        return trade_df

def position_sizing(trade_df, initial_cap, max_risk = 0.01, commission = 0.0005):
    
    max_cap_per_trade = max_risk*initial_cap

    trade_df['stop_diff'] = trade_df['entry_price_adj'] - trade_df['stop_loss']
    trade_df = trade_df[(trade_df['stop_diff'] > 0) & (trade_df['entry_price_adj'] > 0)]  # Ensure valid stop losses

    trade_df['quantity'] = (max_cap_per_trade // trade_df['stop_diff']).astype(int)
    trade_df['max_affordable_qty'] = (initial_cap // trade_df['entry_price_adj']).astype(int)
    trade_df['quantity'] = trade_df[['quantity', 'max_affordable_qty']].min(axis=1)
    trade_df = trade_df[trade_df['quantity'] > 0]
    trade_df['commission_cost'] = trade_df['quantity'] * (trade_df['entry_price_adj'] + trade_df['exit_price_adj']) * commission
    trade_df['net_pl'] = trade_df['pl_adj'] - trade_df['commission_cost']
    trade_df['order_cost'] = (trade_df['entry_price_adj'] * trade_df['quantity']) + trade_df['commission_cost']

    for _, row in trade_df.iterrows():
        
        balance = initial_cap
        balances = []

        if row['order_cost'] <= balance:
            balance -= row['order_cost']
        
        else: 
            print("No money for trades")
        
        balances.append(balance)

    trade_df['balance'] = balances

    return trade_df