import pandas as pd
from data.data_collection import DataCollection
from pathlib import Path

DATA_STORE = Path('../main_data_store_JDKv1.h5')

if __name__ == "__main__":
    # import s&p 500 tickers and change type to list
    stock_tickers = pd.read_csv('data/ticker_list.csv', header=None, usecols=[0], names=['symbols'])
    stock_tickers = list(stock_tickers['symbols'])

    # import chosen etf tickers and change type to list
    etf_tickers = pd.read_csv('data/etf_ticker_list.csv', header=0, usecols=[0], names=['symbols'])
    etf_tickers = list(etf_tickers['symbols'])

    # Get fundamentals
    stock_fundamentals = DataCollection.get_stock_fundamentals(stock_tickers)
    etf_fundamentals = DataCollection.get_etf_fundamentals(etf_tickers)

    # Get historical prices
    stock_prices = DataCollection.get_historical_price(stock_tickers)
    etf_prices = DataCollection.get_historical_price(etf_tickers)

    # Save fundamentals and prices to the data store
    with pd.HDFStore(DATA_STORE) as store:
        store.put('stocks/base_fundamentals', stock_fundamentals)
        store.put('etfs/base_fundamentals', etf_fundamentals)
        store.put('stocks/prices/daily', stock_prices)
        store.put('etfs/prices/daily', etf_prices)