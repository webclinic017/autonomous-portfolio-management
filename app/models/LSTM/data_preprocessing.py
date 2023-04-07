import numpy as np
import pandas as pd
from pathlib import Path


idx = pd.IndexSlice

class Preprocessing:
    def __init__(self, dataset_path):
        self.dataset_path = dataset_path
    
    def load_dataset(self):
        """
        Loads the dataset provided, selects a subset of columns, and assigns names to the index.
        """
        # load the dataset
        dataset = pd.read_hdf(self.dataset_path, 'stocks/prices/daily')
        
        # select a subset of columns
        dataset = dataset.loc[idx[:, '2005-02-15':'2023-02-15'], ['adjusted_close', 'volume', 'RSI', 'MACD', 'MACD_signal', 'MACD_hist', 'BB_upper', 'BB_middle', 'BB_lower']]
        
        # assign names to the index
        dataset.index.names = ['ticker', 'date']
        
        return dataset
    
    def select_most_traded_stocks(self, dataset):
        """
        Selects the most traded stocks based on dollar volume and returns a DataFrame of their daily percentage returns.
        """
        # get the number of unique dates in the dataset
        n_dates = len(dataset.index.unique('date'))
        
        # calculate the dollar volume for each stock and rank them based on dollar volume
        dollar_vol = (dataset.adjusted_close.mul(dataset.volume)
                      .unstack('ticker')
                      .dropna(thresh=int(.95 * n_dates), axis=1)
                      .rank(ascending=False, axis=1)
                      .stack('ticker'))

        # select the 500 most traded stocks based on the mean rank of dollar volume
        most_traded = dollar_vol.groupby(level='ticker').mean().nsmallest(500).index

        dataset = dataset.loc[idx[most_traded, :], 'adjusted_close'].unstack('ticker')

        dataset.index = pd.to_datetime(dataset.index)

        # calculate weekly percentage returns for the most traded stocks
        returns = dataset.resample('W').last().pct_change().dropna(axis=0).sort_index(ascending=False)

        # convert the index to a datetime index
        returns.index = pd.to_datetime(returns.index)

        return returns
    
    def create_sliding_window(self, returns, T):
        """
        Creates a sliding window of length T (52 weeks) for the given DataFrame of returns and returns a new DataFrame.
        """
        # get the number of rows in the returns DataFrame
        n = len(returns)

        # create a list of column names for the sliding window
        tcols = list(range(1, T+1))

        # get the tickers from the returns DataFrame
        tickers = returns.columns

        # create a list of DataFrames for each window
        data_list = [returns.iloc[i:i+T+1].reset_index(drop=True).T
                     .assign(date=returns.index[i], ticker=tickers)
                     .set_index(['ticker', 'date'])
                     for i in range(n-T-1)]

        # concatenate the list of DataFrames into a single DataFrame
        data = pd.concat(data_list)

        # rename the first column to 'fwd_returns' and drop any rows with missing values
        data = data.rename(columns={0: 'fwd_returns'}).sort_index().dropna()

        # clip the values of the remaining columns to the 1st and 99th percentiles
        data.loc[:, tcols] = (data.loc[:, tcols].apply(lambda x: x.clip(lower=x.quantile(.01),
                                                                         upper=x.quantile(.99))))

        # create a new column 'label' indicating whether the forward returns are positive
        data['label'] = (data['fwd_returns'] > 0).astype(int)

        return data
    