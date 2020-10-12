import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import base64
import sys
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.model_selection import train_test_split
import pickle
import plotly.express as px
import plotly.figure_factory as ff
import time

from PIL import Image
import plotly.graph_objects as go


from datetime import datetime
import os 
import matplotlib.pyplot as plt
from alpha_vantage.timeseries import TimeSeries
from datetime import datetime
import talib

from typing import TypeVar, Callable, Sequence
from functools import reduce
T = TypeVar('T')



import glob
from IPython.display import display, HTML
from tqdm import tqdm_notebook as tqdm
from sklearn import preprocessing

import json


import matplotlib.pyplot as plt
from IPython.core.interactiveshell import InteractiveShell
InteractiveShell.ast_node_interactivity = "all"
import os

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
from sklearn import metrics
from sklearn.ensemble import RandomForestRegressor




class Stock_Prediction_Modeling():
    def __init__(self, Stocks, models, features):
        self.Stocks = Stocks
        self.train_Models = models
        self.metrics = {}
        self.features_selected = features
        
        
    def get_stock_data(self, Ticker):
        
        file = self.Ticker + '.csv'
        Stock = pd.read_csv(file,  index_col=0)
        print(Stock)
        #print(self.features_selected)
        print('Loading Historical Price data for ' + self.Ticker + '....')
        
        self.df_Stock = Stock.copy() #[features_selected]
        #self.df_Stock = self.df_Stock.drop(columns=['Date_col'])
        self.df_Stock = self.df_Stock[self.features_selected]
        
        self.df_Stock = self.df_Stock.rename(columns={'Close(t)':'Close'})
        
        #self.df_Stock = self.df_Stock.copy()
        self.df_Stock['Diff'] = self.df_Stock['Close'] - self.df_Stock['Open']
        self.df_Stock['High-low'] = self.df_Stock['High'] - self.df_Stock['Low']
        
        #print('aaaa')
        st.write('Training Selected Machine Learning models for ', self.Ticker)
        st.markdown('Your **_final_ _dataframe_ _for_ Training** ')
        st.write(self.df_Stock)
        my_bar = st.progress(0)
        for percent_complete in range(100):
            time.sleep(0.01)
            my_bar.progress(percent_complete + 1)
        st.success('Training Completed!')

        #self.df_Stock = self.df_Stock[:-70]
        
        print(self.df_Stock.columns)


    def prepare_lagged_features(self, lag_stock, lag_index, lag_diff):

        print('Preparing Lagged Features for Stock, Index Funds.....')
        lags = range(1, lag_stock+1)
        lag_cols= ['Close']
        self.df_Stock=self.df_Stock.assign(**{
            '{}(t-{})'.format(col, l): self.df_Stock[col].shift(l)
            for l in lags
            for col in lag_cols
        })

       
        lags = range(1, lag_index+1)
        lag_cols= ['SPY_Close']
        self.df_Stock= self.df_Stock.assign(**{
            '{}(t-{})'.format(col, l): self.df_Stock[col].shift(l)
            for l in lags
            for col in lag_cols
        })

        self.df_Stock = self.df_Stock.drop(columns=lag_cols)


        lags = range(1, lag_diff+1)
        lag_cols= ['Diff','High-low']
        self.df_Stock= self.df_Stock.assign(**{
            '{}(t-{})'.format(col, l): self.df_Stock[col].shift(l)
            for l in lags
            for col in lag_cols
        })

        self.df_Stock = self.df_Stock.drop(columns=lag_cols)

        remove_lags_na = max(lag_stock, lag_index, lag_diff) + 1
        print('Removing NAN rows - ', str(remove_lags_na))
        self.df_Stock = self.df_Stock.iloc[remove_lags_na:,]
        return self.df_Stock

    def get_lagged_features(self, Ticker):
        
        self.df_Stock_lagged = self.prepare_lagged_features(lag_stock = 20, lag_index = 10, lag_diff = 5)

        print(self.df_Stock_lagged.columns)
        
        self.df_Stock = self.df_Stock_lagged.copy()
        print(self.df_Stock.shape)
        print('Extracted Feature Columns after lagged effect - ')
        print(self.df_Stock.columns)
        
        

    def create_train_test_set(self):

        #self.df_Stock = self.df_Stock[:-60]
        self.features = self.df_Stock.drop(columns=['Close'], axis=1)
        self.target = self.df_Stock['Close']


        data_len = self.df_Stock.shape[0]
        print('Historical Stock Data length is - ', str(data_len))

        #create a chronological split for train and testing
        train_split = int(data_len * 0.9)
        print('Training Set length - ', str(train_split))

        val_split = train_split + int(data_len * 0.08)
        print('Validation Set length - ', str(int(data_len * 0.1)))

        print('Test Set length - ', str(int(data_len * 0.02)))

        # Splitting features and target into train, validation and test samples 
        X_train, X_val, X_test = self.features[:train_split], self.features[train_split:val_split], self.features[val_split:]
        Y_train, Y_val, Y_test = self.target[:train_split], self.target[train_split:val_split], self.target[val_split:]

        #print shape of samples
        print(X_train.shape, X_val.shape, X_test.shape)
        print(Y_train.shape, Y_val.shape, Y_test.shape)

        return X_train, X_val, X_test, Y_train, Y_val, Y_test

    def get_train_test(self):
        print('Splitting the data into Train and Test ...')
        print(' ')
        if self.ML_Model == 'LSTM':
            self.scale_LSTM_features()
            self.X_train, self.X_test, self.Y_train, self.Y_test = self.create_train_test_LSTM()
        else:
            self.X_train, self.X_val, self.X_test, self.Y_train, self.Y_val, self.Y_test = self.create_train_test_set()
            #print('here6')

    def get_mape(self, y_true, y_pred): 
        """
        Compute mean absolute percentage error (MAPE)
        """
        y_true, y_pred = np.array(y_true), np.array(y_pred)
        return np.mean(np.abs((y_true - y_pred) / y_true)) * 100

    
    def calc_metrics(self):
        print('Evaluating Metrics - MAE, MAPE, RMSE, R Square')
        print(' ')
        if self.ML_Model == 'LSTM':
        
            self.Train_RSq = round(metrics.r2_score(self.Y_train,self.Y_train_pred),2)
            self.Train_EV = round(metrics.explained_variance_score(self.Y_train,self.Y_train_pred),2)
            self.Train_MAPE = round(self.get_mape(self.Y_train,self.Y_train_pred), 2)
            self.Train_MSE = round(metrics.mean_squared_error(self.Y_train,self.Y_train_pred), 2) 
            self.Train_RMSE = round(np.sqrt(metrics.mean_squared_error(self.Y_train,self.Y_train_pred)),2)
            self.Train_MAE = round(metrics.mean_absolute_error(self.Y_train,self.Y_train_pred),2)

            
            self.Test_RSq = round(metrics.r2_score(self.Y_test,self.Y_test_pred),2)
            self.Test_EV = round(metrics.explained_variance_score(self.Y_test,self.Y_test_pred),2)
            self.Test_MAPE = round(self.get_mape(self.Y_test,self.Y_test_pred), 2)
            self.Test_MSE = round(metrics.mean_squared_error(self.Y_test,self.Y_test_pred), 2) 
            self.Test_RMSE = round(np.sqrt(metrics.mean_squared_error(self.Y_test,self.Y_test_pred)),2)
            self.Test_MAE = round(metrics.mean_absolute_error(self.Y_test,self.Y_test_pred),2)
        else:
            
            self.Train_RSq = round(metrics.r2_score(self.Y_train,self.Y_train_pred),2)
            self.Train_EV = round(metrics.explained_variance_score(self.Y_train,self.Y_train_pred),2)
            self.Train_MAPE = round(self.get_mape(self.Y_train,self.Y_train_pred), 2)
            self.Train_MSE = round(metrics.mean_squared_error(self.Y_train,self.Y_train_pred), 2) 
            self.Train_RMSE = round(np.sqrt(metrics.mean_squared_error(self.Y_train,self.Y_train_pred)),2)
            self.Train_MAE = round(metrics.mean_absolute_error(self.Y_train,self.Y_train_pred),2)

            self.Val_RSq = round(metrics.r2_score(self.Y_val,self.Y_val_pred),2)
            self.Val_EV = round(metrics.explained_variance_score(self.Y_val,self.Y_val_pred),2)
            self.Val_MAPE = round(self.get_mape(self.Y_val,self.Y_val_pred), 2)
            self.Val_MSE = round(metrics.mean_squared_error(self.Y_train,self.Y_train_pred), 2) 
            self.Val_RMSE = round(np.sqrt(metrics.mean_squared_error(self.Y_val,self.Y_val_pred)),2)
            self.Val_MAE = round(metrics.mean_absolute_error(self.Y_val,self.Y_val_pred),2)

            self.Test_RSq = round(metrics.r2_score(self.Y_test,self.Y_test_pred),2)
            self.Test_EV = round(metrics.explained_variance_score(self.Y_test,self.Y_test_pred),2)
            self.Test_MAPE = round(self.get_mape(self.Y_test,self.Y_test_pred), 2)
            self.Test_MSE = round(metrics.mean_squared_error(self.Y_test,self.Y_test_pred), 2) 
            self.Test_RMSE = round(np.sqrt(metrics.mean_squared_error(self.Y_test,self.Y_test_pred)),2)
            self.Test_MAE = round(metrics.mean_absolute_error(self.Y_test,self.Y_test_pred),2)


    def update_metrics_tracker(self):
        print('Updating the metrics tracker....')
        if self.ML_Model == 'LSTM':
            #self.metrics[self.Ticker] = {}
            self.metrics[self.Ticker][self.ML_Model] = {'Train_MAE': self.Train_MAE, 'Train_MAPE': self.Train_MAPE , 'Train_RMSE': self.Train_RMSE,
                          'Test_MAE': self.Test_MAE, 'Test_MAPE': self.Test_MAPE, 'Test_RMSE': self.Test_RMSE}
        else:
            ##self.metrics[self.Ticker] = {{}}
            self.metrics[self.Ticker][self.ML_Model] = {'Train_MAE': self.Train_MAE, 'Train_MAPE': self.Train_MAPE , 'Train_RMSE': self.Train_RMSE,
                          'Test_MAE': self.Val_MAE, 'Test_MAPE': self.Val_MAPE, 'Test_RMSE': self.Val_RMSE}

       

    def train_model(self, Ticker):

        for model in self.train_Models:
            self.ML_Model = model
            if self.ML_Model == 'Linear Regression':
                
                print(' ')
                print('Training Linear Regressiom Model')
                
                self.get_train_test()
                
                
                from sklearn.linear_model import LinearRegression
                lr = LinearRegression()
                lr.fit(self.X_train, self.Y_train)
                print('LR Coefficients: \n', lr.coef_)
                print('LR Intercept: \n', lr.intercept_)

                print("Performance (R^2): ", lr.score(self.X_train, self.Y_train))

                self.Y_train_pred = lr.predict(self.X_train)
                self.Y_val_pred = lr.predict(self.X_val)
                self.Y_test_pred = lr.predict(self.X_test)

                self.calc_metrics()
                self.update_metrics_tracker()
                self.plot_prediction()
                
            
                
            elif self.ML_Model == 'Random Forest':
                print(' ')
                print('Training Random Forest Model')
                
                self.get_train_test()
                rf = RandomForestRegressor(n_estimators=100, max_depth=50, random_state=42)
                rf.fit(self.X_train, self.Y_train)
                
                self.Y_train_pred = rf.predict(self.X_train)
                self.Y_val_pred = rf.predict(self.X_val)
                self.Y_test_pred = rf.predict(self.X_test)
                
                self.calc_metrics()
                self.update_metrics_tracker()
                self.plot_prediction()
                
    def plot_prediction(self):
        
        print(' ')
        print('Predicted vs Actual for ', self.ML_Model)
        st.write('Predicted vs Actual for ', self.ML_Model)
        self.df_pred = pd.DataFrame(self.Y_val.values, columns=['Actual'], index=self.Y_val.index)
        self.df_pred['Predicted'] = self.Y_val_pred
        self.df_pred = self.df_pred.reset_index()
        self.df_pred.loc[:, 'Date'] = pd.to_datetime(self.df_pred['Date'],format='%Y-%m-%d')
        print('Stock Prediction on Test Data - ',self.df_pred)
        st.write('Stock Prediction on Test Data for - ',self.Ticker)
        st.write(self.df_pred)

        print('Plotting Actual vs Predicted for - ', self.ML_Model)
        st.write('Plotting Actual vs Predicted for - ', self.ML_Model)
        fig = self.df_pred[['Actual', 'Predicted']].plot()
        plt.title('Actual vs Predicted Stock Prices')
        #plt.show()
        #st.write(fig)
        st.pyplot()


    
    def save_results(self, Ticker):
        import json
        print('Saving Metrics in Json for Stock - ', self.Ticker)
        with open('./metrics.txt', 'w') as json_file:
            json.dump(self.metrics, json_file)
        
    
    def pipeline(self,
        value: T,
        function_pipeline: Sequence[Callable[[T], T]],
        ) -> T:
    
        return reduce(lambda v, f: f(v), function_pipeline, value)

    def pipeline_sequence(self):
        for stock in self.Stocks:
            self.Ticker = stock
            self.metrics[self.Ticker] = {}
            print('Initiating Pipeline for Stock Ticker ---- ', self.Ticker)
            st.write('Initiating Pipeline for Stock Ticker ---', self.Ticker)
            z = self.pipeline(
                value=self.Ticker,
                function_pipeline=(
                    self.get_stock_data,
                    self.get_lagged_features,
                    self.train_model, 
                    self.save_results
                        )
                    )

            print(f'z={z}')
