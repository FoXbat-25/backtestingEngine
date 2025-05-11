from config import SQL_POSTGRES_CONN

import psycopg2
import pandas as pd 
from datetime import datetime

def trade_book(nATR, df, strategy):
    with psycopg2.connect(SQL_POSTGRES_CONN) as conn:
        with conn.cursor() as cursor:
            
            entry_insert_query = f"""
                INSERT INTO TRADE_BOOK (symbol, entry_date, entry_price, status, strategy, created_at, updated_at, stop_loss)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
                ON CONFLICT (symbol, entry_date)
                DO NOTHING
            """
            exit_insert_query = f"""
                UPDATE TRADE_BOOK
                SET exit_date = %s, exit_price = %s, updated_at = %s,  status = 'closed'
                WHERE symbol = %s AND status = 'open'
            """
            stop_loss_exit_query = f"""
                UPDATE TRADE_BOOK
                SET exit_date = %s, exit_price = %s, updated_at = %s, status = 'closed', stp_lss_trggrd = True, cooldown_end_date = %s
                WHERE symbol = %s AND status = 'open'
            """

            latest_trade_date = df['prev_date'].max()              
            
            data = df[df['date']<=latest_trade_date]

            symbols = data['symbol'].unique()

            for symbol in symbols:
                
                symbol_df = data[data['symbol'] == symbol].sort_values('date', ascending = True)
                
                cooldown_end_date = None
                open_positions = None
                entry_atr = None

                for _, row in symbol_df.iterrows():

                        if cooldown_end_date is not None and row['date'] >= cooldown_end_date:
                            cooldown_end_date = None                    
                    
                        if open_positions is None and row ['Strong_Buy'] == 1 and cooldown_end_date is None:
                            
                            open_positions = row['next_open']
                            entry_atr = (row['ATR'])

                            stop_loss = (open_positions - (nATR*entry_atr))

                            now_timestamp = datetime.now()

                            entry_trade_data = [row['symbol'],
                                        row['next_date'], 
                                        row['next_open'], 
                                        'open', 
                                        strategy, 
                                        now_timestamp, 
                                        now_timestamp,
                                        stop_loss]
                            
                            cursor.execute(entry_insert_query, tuple(entry_trade_data))
                            
                            

                        elif open_positions is not None and row ['Strong_Sell'] == 1 and row['next_date'] >= pd.to_datetime(open_positions):
                            now_timestamp = datetime.now()

                            exit_trade_data = [row['next_date'],
                                        row['next_open'],
                                        now_timestamp,
                                        row['symbol']]
                            
                            cursor.execute(exit_insert_query, tuple(exit_trade_data))
                            open_positions = None
                            entry_atr = None
                        
                        elif open_positions is not None and (row['close'] <= stop_loss) and row['next_date'] >= pd.to_datetime(open_positions):
                
                            now_timestamp = datetime.now()
                            stop_loss_exit_data = [row['next_date'],
                                            row['next_open'],
                                            now_timestamp,
                                            row['cooldown_end_date'],
                                            row['symbol']
                                            ]
                            cursor.execute(stop_loss_exit_query, tuple(stop_loss_exit_data))
                            open_positions = None
                            entry_atr = None
                            cooldown_end_date = row['cooldown_end_date']
                        
        
        conn.commit()
    

def trade_book2(nATR, df, strategy):

    latest_trade_date = df['prev_date'].max()              
            
    data = df[df['date']<=latest_trade_date]
    trades=[]

    for symbol, group in data.groupby('symbol'):
                
        cooldown_end_date = None
        open_positions = None
        entry_atr = None

        group = group.sort_values('date')

        for _, row in group.iterrows():

            if cooldown_end_date is not None and row['date'] >= cooldown_end_date:
                cooldown_end_date = None                    
        
            if open_positions is None and row ['Strong_Buy'] == 1 and cooldown_end_date is None:
                
                open_positions = row['next_open']
                entry_atr = (row['ATR'])
                now_timestamp = datetime.now()

                stop_loss = (open_positions - (nATR*entry_atr))

                open_trade = {'symbol' : row['symbol'],
                            'entry_date': row['next_date'], 
                            'entry_price': row['next_open'],
                            'exit_date': None,
                            'exit_price': None, 
                            'status': 'open', 
                            'strategy': strategy, 
                            'created_at': now_timestamp, 
                            'updated_at': now_timestamp,
                            'stp_lss_trggrd': False,
                            'cooldown_end_date': None,
                            'stop_loss': stop_loss}
                
                # cursor.execute(entry_insert_query, tuple(entry_trade_data))
                
                

            elif open_positions is not None and row ['Strong_Sell'] == 1 and row['next_date'] >= pd.to_datetime(open_positions):
                now_timestamp = datetime.now()

                open_trade['exit_date'] = row['next_date']
                open_trade['exit_price'] = row['next_open']
                open_trade['updated_at'] = now_timestamp
                open_trade['status'] = 'closed'
                
                # cursor.execute(exit_insert_query, tuple(exit_trade_data))
                trades.append(open_trade)
                open_positions = None
                entry_atr = None
            
            elif open_positions is not None and (row['close'] <= stop_loss) and row['next_date'] >= pd.to_datetime(open_positions):
    
                now_timestamp = datetime.now()
                open_trade['exit_date'] = row['next_date']
                open_trade['exit_price'] = row['next_open']
                open_trade['updated_at'] = now_timestamp
                open_trade['status'] = 'closed'
                open_trade['stp_lss_trggrd'] = True
                open_trade['cooldown_end_date'] = row['cooldown_end_date']
                # cursor.execute(stop_loss_exit_query, tuple(stop_loss_exit_data))

                trades.append(open_trade)
                open_positions = None
                entry_atr = None
                cooldown_end_date = row['cooldown_end_date']

    trades_df = pd.DataFrame(trades)
    return trades_df

