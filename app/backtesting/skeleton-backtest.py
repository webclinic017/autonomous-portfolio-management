import pandas as pd
from pandas import IndexSlice as idx
import backtrader as bt

# Load the dataset
dataset = pd.read_hdf('/home/groovyjac/projects/autonomous-portfolio-management/main_data_store_JDKv1.h5', 'stocks/prices/daily')

# Select a subset of columns
dataset = dataset.loc[idx[:, '2021-02-15':'2023-02-15'], ['adjusted_close', 'volume']]

# Assign names to the index
dataset.index.names = ['ticker', 'date']

# Reset the index
dataset = dataset.reset_index()

# Convert the 'date' column to datetime
dataset['date'] = pd.to_datetime(dataset['date'])

class CustomPandasData(bt.feeds.PandasData):
    lines = ('close', 'volume')
    params = (
        ('datetime', 'date'),
        ('open', 'adjusted_close'),
        ('high', 'adjusted_close'),
        ('low', 'adjusted_close'),
        ('close', 'adjusted_close'),
        ('volume', 'volume'),
        ('openinterest', None)
    )

class CompositeStrategy(bt.Strategy):
    params = (
        ('fast', 10),
        ('slow', 30),
        ('rsi_period', 14),
        ('macd_short', 12),
        ('macd_long', 26),
        ('macd_signal', 9),
        ('bbands_period', 20),
        ('bbands_devfactor', 2),
        ('stop_loss_percent', 0.15),
        ('ticker', 'AAPL')
    )

    def __init__(self):
        # Calculate RSI, MACD, and BBands
        self.rsi = bt.indicators.RSI_SMA(self.data.close, period=self.params.rsi_period)
        self.macd = bt.indicators.MACD(self.data.close, period_me1=self.params.macd_short, period_me2=self.params.macd_long, period_signal=self.params.macd_signal)
        self.bbands = bt.indicators.BollingerBands(self.data.close, period=self.params.bbands_period, devfactor=self.params.bbands_devfactor)
        
        # Calculate Moving Averages for stop-loss mechanism
        self.fast_ma = bt.indicators.SimpleMovingAverage(self.data.close, period=self.params.fast)
        self.slow_ma = bt.indicators.SimpleMovingAverage(self.data.close, period=self.params.slow)
        self.crossover = bt.indicators.CrossOver(self.fast_ma, self.slow_ma)
        self.order = None
        self.stop_loss = None

        # Validate stop-loss percentage
        if self.params.stop_loss_percent < 0 or self.params.stop_loss_percent > 1:
            self.log('Invalid stop-loss percentage, setting to default value of 0.15')
            self.params.stop_loss_percent = 0.15

    def log(self, txt, dt=None):
        ''' Logging function for this strategy '''
        dt = dt or self.data.datetime[0]
        if isinstance(dt, float):
            dt = bt.num2date(dt)
        print('%s, %s' % (dt.date(), txt))

    def notify_order(self, order):
        if order.status in [order.Submitted, order.Accepted]:
            self.log('Order Submitted/Accepted: %s' % order.ref)
            return

        if order.status == order.Completed:
            if order.isbuy():
                self.log('BUY EXECUTED: %s' % order.executed.price)
                self.stop_loss = order.executed.price * (1.0 - self.params.stop_loss_percent)
                self.log('Stop-loss price set to: %s' % self.stop_loss)
            elif order.issell():
                self.log('SELL EXECUTED: %s' % order.executed.price)
                self.stop_loss = None
        elif order.status in [order.Cancelled, order.Margin, order.Rejected]:
            self.log('Order Canceled/Margin/Rejected: %s' % order.ref)
        self.order = None

    def next(self):
        # Generate individual signals for each indicator
        rsi_buy_signal = self.rsi < 30
        rsi_sell_signal = self.rsi > 70
        macd_buy_signal = self.macd.macd > self.macd.signal
        macd_sell_signal = self.macd.macd < self.macd.signal
        bbands_buy_signal = self.data.close < self.bbands.lines.bot
        bbands_sell_signal = self.data.close > self.bbands.lines.top

        # Log the individual signals
        self.log('RSI Buy Signal: {}'.format(rsi_buy_signal))
        self.log('RSI Sell Signal: {}'.format(rsi_sell_signal))
        self.log('MACD Buy Signal: {}'.format(macd_buy_signal))
        self.log('MACD Sell Signal: {}'.format(macd_sell_signal))
        self.log('BBands Buy Signal: {}'.format(bbands_buy_signal))
        self.log('BBands Sell Signal: {}'.format(bbands_sell_signal))

        if self.order:
            return  # Wait for the pending order to complete
        
        if not self.position:
            if rsi_buy_signal or macd_buy_signal or bbands_buy_signal:  # Buy if any buy signal
                self.log('Signal to BUY')
                self.order = self.buy()
        elif self.position:
            if rsi_sell_signal or macd_sell_signal or bbands_sell_signal:  # Sell if any sell signal
                self.log('Signal to SELL (Crossover)')
                self.order = self.close()
            elif self.data.close[0] <= self.stop_loss:
                self.log('Signal to SELL (Stop-loss hit)')
                self.order = self.close()

    def stop(self):
        self.log('Final Portfolio Value: %.2f' % self.broker.getvalue())

# Initialize the Cerebro engine
cerebro = bt.Cerebro()

# Add the composite strategy to the Cerebro engine
cerebro.addstrategy(CompositeStrategy, ticker='AAPL')

# Create a data feed from the dataset
data = CustomPandasData(dataname=dataset[dataset['ticker'] == 'AAPL'])

# Add the data feed to the Cerebro engine
cerebro.adddata(data)

# Set the initial cash balance
cerebro.broker.setcash(10000.0)

# Set the commission
cerebro.broker.setcommission(commission=0.001)

# Add observers for trade and portfolio performance metrics
cerebro.addanalyzer(bt.analyzers.Transactions, _name='transactions_analyzer')
cerebro.addobserver(bt.observers.Broker)
cerebro.addobserver(bt.observers.DrawDown)
cerebro.addobserver(bt.observers.BuySell)

# Print the starting portfolio value
#print('Starting Portfolio Value: %.2f' % cerebro.broker.getvalue())

# Run the backtest and retrieve the results and analyzers
results = cerebro.run()
result = results[0]
transactions_analyzer = result.analyzers.transactions_analyzer

# Analyze the performance
transactions_by_date = transactions_analyzer.get_analysis()
transactions = [t for trans_list in transactions_by_date.values() for t in trans_list]
num_trades = len(transactions)
# num_wins = sum(1 for t in transactions if t.price * t.size > 0)
# num_losses = sum(1 for t in transactions if t.price * t.size < 0)
# winning_rate = num_wins / num_trades * 100 if num_trades > 0 else 0

# Print the performance metrics
print('Starting Portfolio Value: %.2f' % cerebro.broker.startingcash)
print('Final Portfolio Value: %.2f' % cerebro.broker.getvalue())
print('Number of Trades: %d' % num_trades)
#print('Winning Trades: %d' % num_wins)
#print('Losing Trades: %d' % num_losses)
#print('Winning Rate: %.2f%%' % winning_rate)

# add
# pyfolio analysis
# multiple tickers
# PMPT optimization engine
# LSTM
# ensure trade logic is sound