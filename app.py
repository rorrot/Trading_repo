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

################### import de class stock et model2 ######################

from stock import Stocks
from model2 import Stock_Prediction_Modeling

######################## end ################################

def configure_plotly_browser_state():
    import IPython
    display(IPython.core.display.HTML('''
        <script src="/static/components/requirejs/require.js"></script>
        <script>
          requirejs.config({
            paths: {
              base: '/static/base',
              plotly: 'https://cdn.plot.ly/plotly-latest.min.js?noext',
            },
          });
        </script>
        '''))

##### copying pipeline_sequence from model2
# Stocks = ['AAPL', 'GOOG']
models = ['Linear Regression', 'Random Forest']#, 'LSTM']


######################### end #####################################
df_stock = pd.DataFrame()
eval_metrics = {}

st.title("Stock Prediction!")

st.markdown("""
<style>
body {
    color: #fff;
    background-color: #0A3648;

}
</style>
    """, unsafe_allow_html=True)
#0A3648
#13393E
menu=["Stocks Exploration & Feature Extraction", "Train Your Own Machine (Machine Learning Models)","Model (LSTM)"]
choices = st.sidebar.selectbox("Select Dashboard",menu)



if choices == 'Stocks Exploration & Feature Extraction':
    st.subheader('Stock Exploration & Feature extraction')
    st.sidebar.success(" ")

    st.write('Machine Learning algorithm for forcasting Stock Prices.')
    user_input = ''
    st.markdown('Enter **_Ticker_ Symbol** for the **Stock**')
    user_input = st.text_input("", '')
    
    if not user_input:
            pass
    else:
        
        st.markdown('Select from the options below to Explore Stocks')
        
        selected_explore = st.selectbox("", options=['Select your Option', 'Stock Financials Exploration', 'Extract Features for Stock Price Forecasting'], index=0)
        if selected_explore == 'Stock Financials Exploration':
            st.markdown('')
            st.markdown('**_Stock_ Financial** Information')
            st.markdown('')
            st.markdown('')
            stock_financials(user_input)
            plot_time_series(user_input)

    
        elif selected_explore == 'Extract Features for Stock Price Forecasting':
            

            st.markdown('**_Real-Time_ _Feature_ Extraction** for any Stocks')
            
            st.write('Select a Date from a minimum of a year before as some of the features we extract uses upto 200 days of data. ')
            st.markdown('Select **_Start_ _Date_ _for_ _Historical_ Stock** Data & features')
            start_date = st.date_input(
            "", datetime(2015, 1, 4))
            st.write('You selected data from -', start_date)

            submit = st.button('Extract Features')
            if submit:
                try:
                    
                    with st.spinner('Extracting Features... '):
                        time.sleep(2)
                    print('Date - ', start_date)
                    features = Stocks(user_input, start_date, 1)
                    features.pipeline_sequence()

                except:
                    st.markdown('Your **_Ticker_ symbol** should be correct!!! ')
                file_name = user_input + '.csv'
                df_stock = pd.read_csv(file_name)
                st.write('Extracted Features Dataframe for ', user_input)
                st.write(df_stock)
                #st.write('Download Link')

                st.write('We have extracted', len(df_stock.columns), 'columns for this stock. You can Analyse it or even train it for Stock Prediction.')


                st.write('Extracted Feature Columns are', df_stock.columns)

elif choices == 'Train Your Own Machine (Machine Learning Models)':
    st.subheader('Train Machine Learning Models for Stock Prediction & Generate your own Buy/Sell Signals using the best Model')
    st.sidebar.success("")

    
    st.markdown('**_Real_ _Time_ ML Training** for any Stocks')
    st.write('Model training on Multiple stocks at the same time and evaluating them')

    
    st.write('Make sure you have Extracted features for the Stocks you want to train models on using first Tab')
    
    result = glob.glob( '*.csv' )
    #st.write( result )
    stock = []
    for val in result:
        stock.append(val.split('.')[0])
    
    st.markdown('**_Recently_ _Extracted_ Stocks** -')
    st.write(stock[:5])
    cols1 = ['GOOG'] 
    st.markdown('**_Select_ _Stocks_ _to_ Train**')
    Stocks = st.multiselect("", stock, default=cols1)
    
    options = ['Linear Regression', 'Random Forest']
    cols2 = ['Linear Regression', 'Random Forest']
    st.markdown('**_Select_ _Machine_ _Learning_ Algorithms** to Train')
    models = st.multiselect("", options, default=cols2)
    
    
    file = './' + stock[0] + '.csv'
    
    df_stock = pd.read_csv(file)
    df_stock = df_stock.drop(columns=['Date', 'Date_col'])
    #st.write(df_stock.columns)
    st.markdown('Select from your **_Extracted_ features** or use default')
    st.write('Select all Extracted features')
    all_features = st.checkbox('Select all Extracted features')
    cols = ['Open', 'High', 'Low', 'Close(t)', 'Upper_Band', 'MA200', 'ATR', 'ROC', 'SPY_Close']
    if all_features:
        cols = df_stock.columns.tolist()
        cols.pop(len(df_stock.columns)-1)

    features = st.multiselect("", df_stock.columns.tolist(), default=cols)
    
    
    submit = st.button('Train Your Model')
    if submit:
        try:
            training = Stock_Prediction_Modeling(Stocks, models, features)
            training.pipeline_sequence() 
            with open('./metrics.txt') as f:
                eval_metrics = json.load(f)

            

        except:
            st.markdown('There seems to be a error - **_check_ logs**')
            print("Unexpected error:", sys.exc_info())
            print()

    
        Metrics = pd.DataFrame.from_dict({(i,j): eval_metrics[i][j] 
                               for i in eval_metrics.keys() 
                               for j in eval_metrics[i].keys()},
                           orient='index')

        st.write(Metrics)
    
    
    
    
elif choices == '(LSTM)':
    st.subheader('Predict Stock Prices for Any Stocks and Generate Buy/Sell Signals')
    st.sidebar.success("................") 