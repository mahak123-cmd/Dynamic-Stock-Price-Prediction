import numpy as np
import pandas as pd
import yfinance as yf
from keras.models import load_model
import streamlit as st
import matplotlib.pyplot as plt

from prophet.plot import plot_plotly

model = load_model("C:\\Users\\Lovkush\\OneDrive\\Desktop\\STOCK custom data\\STOCK.keras")

st.set_page_config(layout='wide')

images0,image1,images1 = st.columns((2.5,6,2))
image1.image('https://miro.medium.com/max/620/0*dunTLlei47QWR7NR.gif')
alpha, beta, gamma = st.columns((4,5,3))
beta.title('Stock Price Prediction')


alpha1, gamma1 = st.columns(2)

alpha1.markdown('Hello and Welcome to Stock Price Predictor. This WebApp is designed to predict the price of select stocks. To get started, Select a stock from the given list and select the starting date and ending date for the same to view its information.')
gamma1.markdown('Stock price prediction is a very researched entity in the modern day world. It helps companies to raise capital, it helps people generate passive income, stock markets represent the state of the economy of the country and it is widely used soutce for people to invest money in companies with high growth potential')


st.markdown("""=======================================================================================================================================================""")


stock = st.text_input('Enter Stock Symbol', 'GOOG')
start = '2012-01-01'
end = '2023-12-01'

data = yf.download(stock, start, end)

rhs, rh, rhs1 = st.columns((2.5,3,2))
rh.markdown("""### Let's have a look at some raw data""")
rds, rd, rds1 = st.columns((2.5,5,1))
rd.write(data)

graphs1, graphs3 = st.columns(2)
graphs1.markdown("""### Opening Price """)
graphs1.line_chart(data.Open)
graphs3.markdown("""### Volume Price """)
graphs3.line_chart(data.Volume)
graphs1.markdown("""### Closing Price """)
graphs1.line_chart(data.Close)
graphs3.markdown("""### Highest Price """)
graphs3.line_chart(data.High)
graphs1.markdown("""### Lowest Price""")
graphs1.line_chart(data.Low)
graphs3.markdown("""### Adjusted Closing Price """)
graphs3.line_chart(data['Adj Close'])

data_train = pd.DataFrame(data.Close[0: int(len(data) * 0.80)])
data_test = pd.DataFrame(data.Close[int(len(data) * 0.80): len(data)])

from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler(feature_range=(0, 1))

past_100_days = data_train.tail(100)
data_test = pd.concat([past_100_days, data_test], ignore_index=True)
data_test_scale = scaler.fit_transform(data_test)

st.subheader('Price vs MA50')
ma_50_days = data.Close.rolling(50).mean()
fig1, ax1 = plt.subplots(figsize=(8, 6))
ax1.plot(ma_50_days, 'r')
ax1.plot(data.Close, 'g')
st.plotly_chart(fig1)

st.subheader('Price vs MA50 vs MA100')
ma_100_days = data.Close.rolling(100).mean()
fig2, ax2 = plt.subplots(figsize=(8, 6))
ax2.plot(ma_50_days, 'r')
ax2.plot(ma_100_days, 'b')
ax2.plot(data.Close, 'g')
st.plotly_chart(fig2)

st.subheader('Price vs MA100 vs MA200')
ma_200_days = data.Close.rolling(200).mean()
fig3, ax3 = plt.subplots(figsize=(8, 6))
ax3.plot(ma_100_days, 'r')
ax3.plot(ma_200_days, 'b')
ax3.plot(data.Close, 'g')
st.plotly_chart(fig3)

x = []
y = []

for i in range(100, data_test_scale.shape[0]):
    x.append(data_test_scale[i - 100:i])
    y.append(data_test_scale[i, 0])

x, y = np.array(x), np.array(y)

predict = model.predict(x)

scale = 1 / scaler.scale_

predict = predict * scale
y = y * scale

st.subheader('Original Price vs Predicted Price')
fig4= plt.figure(figsize=(12, 8))
plt.plot(predict, 'r', label='Predicted Price')
plt.plot(y, 'g', label='Original Price')
plt.xlabel('Time in days')
plt.ylabel('Price')
plt.legend()
st.plotly_chart(fig4)

