import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import yfinance as yf
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_absolute_error
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor
from sklearn.neighbors import KNeighborsRegressor
from xgboost import XGBRegressor
from keras.models import load_model
import streamlit as st
import datetime
import matplotlib.dates as mdates

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import datetime
import matplotlib.dates as mdates
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_absolute_error


# Function to download data
def download_data(ticker, start, end):
    return yf.download(ticker, start=start, end=end)

# Function to display the most recent data
def display_most_recent_data(df):
    most_recent_date = df.index[-1]
    most_recent_data = df.loc[most_recent_date]

    st.markdown(f"""
        <div style="background-color: #f9f9f9; padding: 10px; border-radius: 5px;">
            <h3 style="text-align: center; color: #333;">Most Recent Values on {most_recent_date.date()}</h3>
            <div style="display: flex; justify-content: space-around;">
                <div style="text-align: center;">
                    <p style="margin: 0; font-size: 1.2em; color: #ff6347;">Open</p>
                    <p style="margin: 0; font-size: 1.5em;">{most_recent_data['Open']:.2f}</p>
                </div>
                <div style="text-align: center;">
                    <p style="margin: 0; font-size: 1.2em; color: #4682b4;">High</p>
                    <p style="margin: 0; font-size: 1.5em;">{most_recent_data['High']:.2f}</p>
                </div>
                <div style="text-align: center;">
                    <p style="margin: 0; font-size: 1.2em; color: #32cd32;">Low</p>
                    <p style="margin: 0; font-size: 1.5em;">{most_recent_data['Low']:.2f}</p>
                </div>
                <div style="text-align: center;">
                    <p style="margin: 0; font-size: 1.2em; color: #ff4500;">Close</p>
                    <p style="margin: 0; font-size: 1.5em;">{most_recent_data['Close']:.2f}</p>
                </div>
            </div>
        </div>
    """, unsafe_allow_html=True)

# Function to display technical indicators
def tech_indicators(df, option, start_date, end_date):
    st.subheader('Data from %s to %s' % (start_date, end_date))
    st.write(df.describe())

    # Visualization - Closing Price vs Time
    st.subheader('Closing Price VS Time chart')
    fig = plt.figure(figsize=(12, 6))
    plt.plot(df['Close'])
    plt.xlabel('Date')
    plt.ylabel('Closing Price')
    plt.title(f'Closing Price of {option} from {start_date} to {end_date}')
    st.pyplot(fig)


# Function to display recent 10 days data
def recent_data(df):
    st.subheader('Recent 10 Days Data')
    recent_10_days = df.tail(10)
    st.write(recent_10_days)

# Function for prediction
def predict(df, option):
    model_type = st.radio('Choose a model', ['LinearRegression', 'RandomForestRegressor', 'ExtraTreesRegressor', 'KNeighborsRegressor', 'XGBoostRegressor'])
    num_days = st.number_input('How many days forecast?', value=5, min_value=1)
    num_days = int(num_days)
    
    if st.button('Predict'):
        # Select model
        if model_type == 'LinearRegression':
            model = LinearRegression()
        elif model_type == 'RandomForestRegressor':
            model = RandomForestRegressor()
        elif model_type == 'ExtraTreesRegressor':
            model = ExtraTreesRegressor()
        elif model_type == 'KNeighborsRegressor':
            model = KNeighborsRegressor()
        else:
            model = XGBRegressor()
        
        # Prepare data
        data = df[['Close']].values
        scaler = MinMaxScaler(feature_range=(0, 1))
        data_scaled = scaler.fit_transform(data)
        
        x_train, y_train = [], []
        window_size = 100  # Adjust window size as needed
        for i in range(window_size, len(data_scaled) - num_days):
            x_train.append(data_scaled[i - window_size:i, 0])
            y_train.append(data_scaled[i + num_days, 0])
        x_train, y_train = np.array(x_train), np.array(y_train)
        
        model.fit(x_train, y_train)
        
        # Forecast
        forecast = []
        last_window = data_scaled[-window_size:, 0]
        for _ in range(num_days):
            pred = model.predict(last_window.reshape(1, -1))[0]
            forecast.append(pred)
            last_window = np.roll(last_window, -1)
            last_window[-1] = pred
        
        # Inverse transform to get actual prices
        forecast_prices = scaler.inverse_transform(np.array(forecast).reshape(-1, 1))
        
        # Dates for forecast
        forecast_dates = pd.date_range(df.index[-1] + pd.Timedelta(days=1), periods=num_days, freq='D')
        
        # Display results
        st.subheader(f'Forecasted Closing Price for {option}')
        for i in range(num_days):
            st.write(f"Day {i + 1}: {forecast_prices[i][0]:.2f} on {forecast_dates[i].date()}")
        
        # Plotting
        fig, ax = plt.subplots(figsize=(12, 6))
        ax.plot(df.index, df['Close'], label='Historical Prices')
        ax.plot(forecast_dates, forecast_prices, label='Forecasted Prices')
        ax.set_xlabel('Date')
        ax.set_ylabel('Closing Price')
        ax.set_title(f'Forecasted Closing Price of {option}')
        ax.legend()
        ax.xaxis.set_major_locator(mdates.MonthLocator())
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%b %d'))
        plt.xticks(rotation=45)
        st.pyplot(fig)



def main():
    st.markdown("""
        <style>
            /* Add your own CSS styling here */
        </style>
    """, unsafe_allow_html=True)

    st.title('Stock Trend Prediction')

    option = st.sidebar.text_input('Enter a Stock Symbol', value='AAPL')
    option = option.upper()
    today = datetime.date.today()
    start_date = st.sidebar.date_input('Start Date', value=today - datetime.timedelta(days=365))
    end_date = st.sidebar.date_input('End Date', value=today)

    if start_date < end_date:
        df = download_data(option, start_date, end_date)
        display_most_recent_data(df)
        st.sidebar.success(f'Start date: `{start_date}`\n\nEnd date: `{end_date}`')

        st.sidebar.title('Menu')
        menu_option = st.sidebar.selectbox('Make a choice', ['Visualize', 'Recent Data', 'Predict'])

        if menu_option == 'Visualize':
            tech_indicators(df, option, start_date, end_date)
        elif menu_option == 'Recent Data':
            recent_data(df)
        else:
            predict(df, option)
    else:
        st.sidebar.error('Error: End date must fall after start date')

if __name__ == '__main__':
    main()

