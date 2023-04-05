import warnings
import numpy as np
import talib
import pandas as pd
from pathlib import Path
import requests
from dotenv import load_dotenv
import os
from progressbar import ProgressBar

warnings.filterwarnings('ignore')
pd.set_option('display.expand_frame_repr', False)

load_dotenv()
EOD_TOKEN = os.getenv('EOD_TOKEN')

DATA_STORE = Path('../../main_data_store_JDKv1.h5')


class DataCollection:

    @staticmethod
    def calculate_technical_indicators(df):
        """
        Calculate RSI, MACD, Bollinger Bands, and VWAP for the given DataFrame.

        :param df: DataFrame with OHLCV data
        :return: DataFrame with calculated technical indicators
        """
        # RSI
        df['RSI'] = talib.RSI(df['close'], timeperiod=14)

        # MACD
        macd, macdsignal, macdhist = talib.MACD(df['adjusted_close'], fastperiod=12, slowperiod=26, signalperiod=9)
        df['MACD'] = macd
        df['MACD_signal'] = macdsignal
        df['MACD_hist'] = macdhist

        # Bollinger Bands
        upper, middle, lower = talib.BBANDS(df['adjusted_close'], timeperiod=20)
        df['BB_upper'] = upper
        df['BB_middle'] = middle
        df['BB_lower'] = lower

        # VWAP
        df['VWAP'] = np.cumsum(df['volume'] * (df['high'] + df['low'] + df['close']) / 3) / np.cumsum(df['volume'])

        return df

    @staticmethod
    def get_historical_price(tickers):
        """
        Retrieve historical daily OHLCV prices and volume for a list of tickers.

        :param tickers: List of stock or ETF tickers
        :param data_type: Type of data ('eod' for end-of-day)
        :return: DataFrame with historical prices and technical indicators
        """
        d = {}
        pbar = ProgressBar()

        for i, ticker in pbar(enumerate(tickers)): 

            r = requests.get('https://eodhistoricaldata.com/api' + '/' + 'eod' + '/' + ticker + '.US', 
                params={'api_token': EOD_TOKEN, 'fmt': 'json'}
                )
            data = r.json()
            r.close()

            d[ticker] = DataCollection.calculate_technical_indicators(
                pd.DataFrame.from_records(data).set_index('date')
            )

        df = pd.concat(d.values(), axis=0, keys=d.keys())

        return df

    @staticmethod
    def get_stock_fundamentals(tickers):
        """
        Retrieve stock fundamentals data for a list of tickers.

        :param tickers: List of stock tickers
        :return: Series with stock fundamentals data
        """
        raw_data = {}
        multi_ticker_dict = {}

        pbar = ProgressBar()

        for i, ticker in pbar(enumerate(tickers)): 

            r = requests.get('https://eodhistoricaldata.com/api/fundamentals/' + ticker + '.US', 
                params={'api_token': EOD_TOKEN, 'fmt': 'json'}
                )
            data = r.json()  
            r.close()
            raw_data[ticker] = data

            # Remove unnecessary data
            Officers = raw_data[ticker]['General'].pop('Officers', None)
            Listings = raw_data[ticker]['General'].pop('Listings', None)
            AddressData = raw_data[ticker]['General'].pop('AddressData', None)
            NumberDividendsByYear = raw_data[ticker]['SplitsDividends'].pop('NumberDividendsByYear', None)

            columns = ['General', 'Highlights', 'Valuation', 'SharesStats',
                'Technicals','SplitsDividends']

            single_ticker_dict = {}
            
            for column in columns:
                single_ticker_dict[column] = pd.Series(raw_data[ticker][column])

            single_ticker_series = pd.concat(single_ticker_dict)
            multi_ticker_dict[ticker] = single_ticker_series

        multi_ticker_series = pd.concat(multi_ticker_dict)

        return multi_ticker_series

    @staticmethod
    def get_etf_fundamentals(tickers):
        """
        Retrieve ETF fundamentals data for a list of tickers.

        :param tickers: List of ETF tickers
        :return: Series with ETF fundamentals data
        """
        raw_data = {}
        multi_ticker_dict = {}

        columns = ['ISIN', 'Company_Name', 'Company_URL', 'ETF_URL', 'Domicile',
            'Index_Name', 'Yield', 'Dividend_Paying_Frequency', 'Inception_Date',
            'Max_Annual_Mgmt_Charge', 'Ongoing_Charge', 'Date_Ongoing_Charge',
            'NetExpenseRatio', 'AnnualHoldingsTurnover', 'TotalAssets', 'Holdings_Count',
            'Average_Mkt_Cap_Mil']
        
        pbar = ProgressBar()

        for i, ticker in pbar(enumerate(tickers)): 
            
            r = requests.get('https://eodhistoricaldata.com/api/fundamentals/' + ticker + '.US', 
                params={'api_token': EOD_TOKEN, 'fmt': 'json'}
                )
            data = r.json()
            r.close()
            raw_data[ticker] = data  

            single_ticker_dict = {}

            single_ticker_dict['General'] = pd.Series(raw_data[ticker]['General'])
            single_ticker_dict['Technicals'] = pd.Series(raw_data[ticker]['Technicals'])

            single_ticker_dict['ETF_Data'] = pd.Series([raw_data[ticker]['ETF_Data'][name] for name in columns],
                index=[name for name in columns])
            
            single_ticker_dict['Market_Capitalisation'] = pd.Series(raw_data[ticker]['ETF_Data']['Market_Capitalisation'])
            single_ticker_dict['MorningStar'] = pd.Series(raw_data[ticker]['ETF_Data']['MorningStar'])
            single_ticker_dict['Performance'] = pd.Series(raw_data[ticker]['ETF_Data']['Performance'])

            single_ticker_series = pd.concat(single_ticker_dict)
            multi_ticker_dict[ticker] = single_ticker_series

        multi_ticker_series = pd.concat(multi_ticker_dict)

        return multi_ticker_series

                                                       
                                                       
                                                       
                                                       
                                                       
                                                       
                                                


