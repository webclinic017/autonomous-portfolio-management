import pandas as pd
import numpy as np
import tensorflow as tf
from pathlib import Path
from tensorflow.keras.models import Model
from tensorflow.keras.layers import LSTM, Dense, Input, Embedding, Reshape, BatchNormalization, concatenate
import tensorflow.keras.backend as K
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping

import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import spearmanr
from matplotlib.ticker import FuncFormatter


idx = pd.IndexSlice

class LSTM_model:
    def __init__(self, data, window_size=52, lstm1_units=25, lstm2_units=10, embedding_dim=5):
        self.data = data
        self.window_size = window_size
        self.lstm1_units = lstm1_units
        self.lstm2_units = lstm2_units
        self.embedding_dim = embedding_dim
        self.model = None
        self.results_path = 'results/lstm_model'
    
    def prepare_data(self):
        """
        Prepare input data for the LSTM model and train-test split.
        """
        data = self.data.copy()
        data['ticker'] = pd.factorize(data.index.get_level_values('ticker'))[0]
        data['month'] = data.index.get_level_values('date').month
        data = pd.get_dummies(data, columns=['month'], prefix='month')
        
        window_size = self.window_size
        sequence = list(range(1, window_size+1))
        ticker = 1
        months = 12
        n_tickers = data.ticker.nunique()
        
        train_data = data.loc[idx[:, :'2020'], :]
        test_data = data.loc[idx[:, '2021':],:]
        
        X_train = [
            train_data.loc[:, sequence].values.reshape(-1, window_size , 1),
            train_data.ticker,
            train_data.filter(like='month')
        ]
        y_train = train_data.fwd_returns
        
        X_test = [
            test_data.loc[:, list(range(1, window_size+1))].values.reshape(-1, window_size , 1),
            test_data.ticker,
            test_data.filter(like='month')
        ]
        y_test = test_data.fwd_returns
        
        return X_train, y_train, X_test, y_test
    
    def create_model(self):
        """
        Create the LSTM model architecture.
        """
        K.clear_session()
        n_features = 1
        window_size = self.window_size
        n_tickers = self.data.index.get_level_values(0).nunique()
        
        # Input Layers
        returns = Input(shape=(window_size, n_features), name='Returns')
        tickers = Input(shape=(1,), name='Tickers')
        months = Input(shape=(12,), name='Months')
        
        # LSTM Layers
        lstm1 = LSTM(units=self.lstm1_units, 
                     input_shape=(window_size, n_features), 
                     name='LSTM1', 
                     dropout=.2,
                     return_sequences=True)(returns)
        lstm_model = LSTM(units=self.lstm2_units, 
                          dropout=.2,
                          name='LSTM2')(lstm1)
        
        # Embedding Layer
        ticker_embedding = Embedding(input_dim=n_tickers, 
                                     output_dim=self.embedding_dim, 
                                     input_length=1)(tickers)
        ticker_embedding = Reshape(target_shape=(self.embedding_dim,))(ticker_embedding)
        
        # Concatenate Model components
        merged = concatenate([lstm_model, ticker_embedding, months], name='Merged')
        bn = BatchNormalization()(merged)
        hidden_dense = Dense(10, name='FC1')(bn)
        output = Dense(1, name='Output')(hidden_dense)
        
        # Create and compile the model
        self.model = Model(inputs=[returns, tickers, months], outputs=output)
        print(self.model.summary())
        self.model.compile(loss='mse', optimizer=tf.keras.optimizers.Adam())

    def train_model(self, X_train, y_train, X_test, y_test, epochs=50, batch_size=64, save_best_only=True, early_stopping_patience=5):
        """
        Train the LSTM model using the prepared input data.
        """
        # Create directory to save model if it does not exist
        Path(self.results_path).mkdir(parents=True, exist_ok=True)
        
        # Filepath to save the best model
        lstm_path = f'{self.results_path}/lstm.regression.h5'

        # Model checkpoint callback to save the best model based on validation loss
        checkpointer = ModelCheckpoint(filepath=lstm_path,
                                    verbose=1,
                                    monitor='val_loss',
                                    mode='min',
                                    save_best_only=save_best_only)

        # Early stopping callback to stop training if validation loss does not improve after certain number of epochs
        early_stopping = EarlyStopping(monitor='val_loss', 
                                    patience=early_stopping_patience,
                                    restore_best_weights=True)

        # Train the model
        training = self.model.fit(X_train,
                                y_train,
                                epochs=epochs,
                                batch_size=batch_size,
                                validation_data=(X_test, y_test),
                                callbacks=[early_stopping, checkpointer],
                                verbose=1)
        
        return training

    # def evaluate_model(self, X_test, y_test):
    #     """
    #     Evaluate the model on the test dataset and return predictions.
    #     """
    #     # Load the best model saved during training
    #     self.model.load_weights(f'{self.results_path}/lstm.regression.h5')

    #     # Get model predictions on the test dataset
    #     test_predict = pd.Series(self.model.predict(X_test).squeeze(), index=y_test.index)

    #     # Create DataFrame with actual returns and predictions
    #     df = y_test.to_frame('ret').assign(y_pred=test_predict)
    #     by_date = df.groupby(level='date')
    #     df['deciles'] = by_date.y_pred.apply(pd.qcut, q=5, labels=False, duplicates='drop')
    #     ic = by_date.apply(lambda x: spearmanr(x.ret, x.y_pred)[0]).mul(100)

    #     # Save predictions to hdf file
    #     test_predict = test_predict.to_frame('prediction')
    #     test_predict.index.names = ['symbol', 'date']
    #     test_predict.to_hdf(self.results_path + '/predictions.h5', 'predictions')

    #     # Calculate Spearman rank correlation coefficient
    #     rho, p = spearmanr(df.ret, df.y_pred)

    #     # Plot weekly forward returns by predicted quintile and 4-week rolling IC
    #     fig, axes = plt.subplots(ncols=2, figsize=(14, 4))
    #     sns.barplot(x='deciles', y='ret', data=df, ax=axes[0])
    #     axes[0].set_title('Weekly Fwd Returns by Predicted Quintile')
    #     axes[0].yaxis.set_major_formatter(FuncFormatter(lambda y, _: '{:.2%}'.format(y)))
    #     axes[0].set_ylabel('Weekly Returns')
    #     axes[0].set_xlabel('Quintiles')

    #     avg_ic = ic.mean()
    #     title = f'4-Week Rolling IC | Weekly avg: {avg_ic:.2f} | Overall: {rho*100:.2f}'
    #     ic.rolling(4).mean().dropna().plot(ax=axes[1], title=title)
    #     axes[1].axhline(avg_ic, ls='--', c='k', lw=1)
    #     axes[1].axhline(0, c='k', lw=1)
    #     axes[1].set_ylabel('IC')
    #     axes[1].set_xlabel('Date')

    #     sns.despine()
    #     fig.tight_layout()

    #     # Save the plots as .png files
    #     fig.savefig(self.results_path + '/lstm_reg.png')

    #     return test_predict