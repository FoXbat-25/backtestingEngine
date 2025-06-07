import pandas as pd
import numpy as np
from sqlalchemy import create_engine
from config import SQL_ALCHEMY_CONN

engine = create_engine(SQL_ALCHEMY_CONN)

def main(first_trade_date, symbol="^NSEI"):
    indexes_query = """
        SELECT * FROM INDEXES_FACT 
        WHERE SYMBOL = %(symbol)s AND date >= %(start_date)s;
    """
    indexes_df = pd.read_sql(indexes_query, engine, params={
        "symbol": symbol,
        "start_date": first_trade_date
    })

    indexes_df.to_csv('indexes.csv')

if __name__ == '__main__':
    main('2020-01-01')