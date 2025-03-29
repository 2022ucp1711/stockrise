import numpy as np
import pandas as pd
import yfinance as yf
from keras.models import load_model
import streamlit as st
import matplotlib.pyplot as plt

# Fix 1: Use raw string for model path
model = load_model(r'C:\Python\Stock\Stock Predictions Model.keras')  # Raw string

st.header('Stock Market Predictor')

stock = st.text_input('Enter Stock Symbol', 'GOOG')
start = '2012-01-01'
end = '2022-12-31'

data = yf.download(stock, start, end)

st.subheader('Stock Data')
st.write(data)

# Split data
train_data = data.iloc[:int(len(data)*0.80)]
test_data = data.iloc[int(len(data)*0.80):]

# Fix 4: Fit scaler ONLY on training data
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler(feature_range=(0, 1))
scaler.fit(train_data[['Close']])  # Fit on training data

# Prepare test data with past 100 days
past_100_days = train_data.tail(100)
test_data = pd.concat([past_100_days, test_data], axis=0)
test_data_scaled = scaler.transform(test_data[['Close']])  # Transform (not fit_transform)

# Create sequences
x = []
y = []
for i in range(100, len(test_data_scaled)):
    x.append(test_data_scaled[i-100:i, 0])
    y.append(test_data_scaled[i, 0])

x, y = np.array(x), np.array(y)
x = np.reshape(x, (x.shape[0], x.shape[1], 1))  # Reshape for LSTM input (samples, timesteps, features)

# Predictions
predictions = model.predict(x)
predictions = scaler.inverse_transform(predictions)  # Fix 3: Correct inverse scaling
y_actual = scaler.inverse_transform(y.reshape(-1, 1))  # Fix 3: Inverse scale actual values

# Plotting with explicit figures
st.subheader('Price vs MA50')
fig1, ax1 = plt.subplots(figsize=(8, 6))
ax1.plot(data['Close'], 'g', label='Price')
ax1.plot(data['Close'].rolling(50).mean(), 'r', label='MA50')
ax1.legend()
st.pyplot(fig1)

st.subheader('Price vs MA50 vs MA100')
fig2, ax2 = plt.subplots(figsize=(8, 6))
ax2.plot(data['Close'], 'g', label='Price')
ax2.plot(data['Close'].rolling(50).mean(), 'r', label='MA50')
ax2.plot(data['Close'].rolling(100).mean(), 'b', label='MA100')
ax2.legend()
st.pyplot(fig2)

st.subheader('Original Price vs Predicted Price')
fig4, ax4 = plt.subplots(figsize=(8, 6))
ax4.plot(y_actual, 'g', label='Actual Price')  # Fix 5: Correct labels
ax4.plot(predictions, 'r', label='Predicted Price')
ax4.set_xlabel('Time')
ax4.set_ylabel('Price')
ax4.legend()
st.pyplot(fig4)
