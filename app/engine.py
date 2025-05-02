import pandas as pd
import numpy as np
from sqlalchemy import create_engine

from datetime import datetime

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


    def backtest_engine(self, symbol, strategy_name, risk_free_rate = 0.07):
        
        query="""
            SELECT * 
            FROM TRADE_BOOK WHERE symbol = %(symbol)s and strategy = %(strategy_name)s and status = 'closed';
        """
        trade_df = pd.read_sql(query, engine, params={"symbol": symbol, "strategy": strategy_name})

        trade_df['date'] = pd.to_datetime(trade_df['date'])
        trade_df = trade_df.sort_values(by='date', ascending= True)

        trade_df['daily_return'] = trade_df['pl']/trade_df['holding_period']
        
        trade_df['daily_return%'] = (trade_df['daily_return']/trade_df['entry_price'])

        trade_df['cumulative_return'] = trade_df['daily_return%'].cumsum()

        # Compute drawdown
        trade_df['cumulative_max'] = trade_df['cumulative_return'].cummax()
        trade_df['drawdown'] = trade_df['cumulative_return'] - trade_df['cumulative_max']
        max_drawdown = trade_df['drawdown'].min()

        mean_daily_return = trade_df['daily_return%'].dropna().mean()  
        annual_return = mean_daily_return*250

        std_dev_daily_return = trade_df['daily_return%'].dropna().std()
        annual_volatility = std_dev_daily_return * np.sqrt(250)

        if annual_volatility == 0 or np.isnan(annual_volatility):
            sharpe_ratio = np.nan
        else:
            sharpe_ratio = (annual_return - risk_free_rate)/annual_volatility

        wins = len(trade_df[trade_df['pl'] > 0])
        losses = len(trade_df[trade_df['pl'] < 0])
        total_trades = len(trade_df)
        win_rate = len(wins) / total_trades if total_trades else np.nan
        avg_win = wins['pl'].mean() if not wins.empty else 0
        avg_loss = losses['pl'].mean() if not losses.empty else 0
        profit_factor = wins['pl'].sum() / abs(losses['pl'].sum()) if not losses.empty else np.inf
        payoff_ratio = avg_win / abs(avg_loss) if avg_loss != 0 else np.inf
        expectancy = (win_rate * avg_win) + ((1 - win_rate) * avg_loss)

        return {
            "annual_return": annual_return,
            "annual_volatility": annual_volatility,
            "sharpe_ratio": sharpe_ratio,
            "cumulative_return": trade_df['cumulative_return'].iloc[-1],
            "max_drawdown": max_drawdown,
            "win_rate": win_rate,
            "average_win": avg_win,
            "average_loss": avg_loss,
            "profit_factor": profit_factor,
            "payoff_ratio": payoff_ratio,
            "expectancy": expectancy,
            "total_trades": total_trades
        }