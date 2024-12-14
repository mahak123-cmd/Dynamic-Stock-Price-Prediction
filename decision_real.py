# Importing all necessary libraries.
import numpy as np
import pandas as pd
import yfinance as yf
from keras.models import load_model
import streamlit as st
import matplotlib.pyplot as plt
import joblib

from datetime import date
from prophet import Prophet
from plotly import graph_objs as go
from prophet.plot import plot_plotly

st.set_page_config(layout="wide")
alpha, beta, gamma = st.columns((4,5,3))
beta.title('Stock Price Prediction')
images0,image1,images1 = st.columns((2.5,6,2))
image1.image('https://miro.medium.com/max/620/0*dunTLlei47QWR7NR.gif')
alpha1, gamma1 = st.columns(2)

alpha1.markdown('Hello and Welcome to Stock Price Predictor. This WebApp is designed to predict the price of select stocks. To get started, Select a stock from the given list and select the starting date and ending date for the same to view its information.')
stock = gamma1.markdown('Stock price prediction is a very researched entity in the modern day world. It helps companies to raise capital, it helps people generate passive income, stock markets represent the state of the economy of the country and it is widely used soutce for people to invest money in companies with high growth potential')

stock=st.text_input("Enter stock symbol",'GOOG')
start='2012-01-01'
end = '2023-12-08'
df = yf.download(stock,start,end)

st.write("\n")
st.write("\n")
rhs, rh, rhs1 = st.columns((3,3,2))
rh.markdown("### Let's have a look at raw data")
rds, rd, rds1 = st.columns((2.5,5,1))
rd.write(df)

st.markdown("""======================================================================================================================================================================""")

rhs, rh, rhs1 = st.columns((2.5,3,2))
rh.markdown("""### Let's see all data in Graph form""")
st.write("\n")
st.write("\n")

graphs1, graphs3 = st.columns(2)
graphs1.markdown("""### Opening Price """)
graphs1.line_chart(df.Open)
graphs3.markdown("""### Volume Price """)
graphs3.line_chart(df.Volume)
graphs1.markdown("""### Closing Price """)
graphs1.line_chart(df.Close)
graphs3.markdown("""### Highest Price """)
graphs3.line_chart(df.High)
graphs1.markdown("""### Lowest Price""")
graphs1.line_chart(df.Low)
graphs3.markdown("""### Adjusted Closing Price """)
graphs3.line_chart(df['Adj Close'])

st.markdown("""======================================================================================================================================================================""")
st.title('K-NEAREST NEIGHBOUR :-')
st.write('\n')

df['Open - Close']=df['Open'] - df['Close']
df['High - Low'] = df['High'] - df['Low']

X = df[['Open - Close', 'High - Low']]

Y= np.where(df['Close'].shift(-1)>df['Close'],1,-1)

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, Y ,test_size=0.25, random_state=44)

from sklearn.neighbors import KNeighborsClassifier
from sklearn import neighbors
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score

#using gridsearch to find the best parameter
params = {'n_neighbors': [2,3,4,5,6,7,8,9,10,11,12,13,14,15]}
knn = neighbors.KNeighborsClassifier()
model = GridSearchCV(knn, params, cv=5)

#fit the model
model.fit(X_train, y_train)

#Accuracy Score
accuracy_train = accuracy_score(y_train, model.predict(X_train))
accuracy_test = accuracy_score(y_test, model.predict(X_test))

rhs, rh, rhs1 = st.columns((2.5,3,2))
rhs.subheader('Train Data Accuracy :  %.2f' %accuracy_train)
rhs1.subheader('Test Data Accuracy :  %.2f' %accuracy_test)
st.write('\n')
st.write('\n')

predictions_classification = model.predict(X_test)

actual_predicted_data = pd.DataFrame({'Actual Class':y_test, 'Predicted Class':predictions_classification})
actual_predicted_data.head(10)
x, y, z = st.columns((2.7,3,4))
x.markdown("""### KNN CLASSIFIER""")
z.markdown("""### KNN REGRESSION""")
p, q, r = st.columns((1.3,5,3))
q.write(actual_predicted_data)

y = df['Close']


from sklearn.neighbors import KNeighborsRegressor
from sklearn import neighbors

X_train_reg, X_test_reg, y_train_reg, y_test_reg = train_test_split(X, y, test_size=0.25, random_state=44)

#using gridsearch to find the best parameter
params = {'n_neighbors': [2,3,4,5,6,7,8,9,10,11,12,13,14,15]}
knn_reg = neighbors.KNeighborsRegressor()
model_reg = GridSearchCV(knn_reg, params, cv=5)

#fit the model and make predictions
model_reg.fit(X_train_reg, y_train_reg)
predictions = model_reg.predict(X_test_reg)

valid = pd.DataFrame({'Actual Price':y_test_reg, 'Predicted Price value':predictions})

r.write(valid)

#rmse
rms=np.sqrt(np.mean(np.power((np.array(predictions)),2)))
st.write('RMSE : ',rms)



st.markdown("""======================================================================================================================================================================""")

model = load_model("C:\\Users\\Lovkush\\OneDrive\\Desktop\\STOCK custom data\\STOCK.keras")
st.title('LONG-SHORT TERM MEMORY (LSTM) :-')
st.write('\n')
data_train = pd.DataFrame(df.Close[0: int(len(df) * 0.80)])
data_test = pd.DataFrame(df.Close[int(len(df) * 0.80): len(df)])

from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler(feature_range=(0, 1))

past_100_days = data_train.tail(100)
data_test = pd.concat([past_100_days, data_test], ignore_index=True)
data_test_scale = scaler.fit_transform(data_test)

st.subheader('Price vs MA50')
ma_50_days = df.Close.rolling(50).mean()
fig1, ax1 = plt.subplots(figsize=(13, 6))
ax1.plot(ma_50_days, 'y')
ax1.plot(df.Close, 'g')
st.plotly_chart(fig1)

st.subheader('Price vs MA50 vs MA100')
ma_100_days = df.Close.rolling(100).mean()
fig2, ax2 = plt.subplots(figsize=(13, 6))
ax2.plot(ma_50_days, 'y')
ax2.plot(ma_100_days, 'b')
ax2.plot(df.Close, 'g')
st.plotly_chart(fig2)

st.subheader('Price vs MA100 vs MA200')
ma_200_days = df.Close.rolling(200).mean()
fig3, ax3 = plt.subplots(figsize=(13, 6))
ax3.plot(ma_100_days, 'b')
ax3.plot(ma_200_days, 'orange')
ax3.plot(df.Close, 'g')
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


fig4, ax4 = plt.subplots(figsize=(13, 6))
ax4.plot(predict, 'r', label='Predicted Price')
ax4.plot(y, 'g', label='Original Price')
plt.xlabel('Time in days')
plt.ylabel('Price')
plt.legend()
st.plotly_chart(fig4)

st.markdown("""======================================================================================================================================================================""")

df2 = df['Close']
df2 = pd.DataFrame(df2) 

loaded_tree = joblib.load("C:\\Users\\Lovkush\\OneDrive\\Desktop\\STOCK custom data\\Decision Tree Custom\\decision_tree_model.joblib")
# Prediction 100 days into the future.
future_days = 100
df2['Prediction'] = df2['Close'].shift(-future_days)
df2.tail()

#LINEAR AND DECISION TREE REGRESSION

X = np.array(df2.drop(['Prediction'], axis=1))[:-future_days]
y = np.array(df2['Prediction'])[:-future_days]

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(X, y, test_size = 0.2)

from sklearn.tree import DecisionTreeRegressor
from sklearn.linear_model import LinearRegression

# Implementing Linear and Decision Tree Regression Algorithms.
tree = DecisionTreeRegressor().fit(x_train, y_train)
lr = LinearRegression().fit(x_train, y_train)

x_future = df2.drop(['Prediction'], axis=1)[:-future_days]
x_future = x_future.tail(future_days)
x_future = np.array(x_future)

tree_prediction = tree.predict(x_future)
lr_prediction = lr.predict(x_future)
predictions = tree_prediction 
valid = df2[X.shape[0]:]
valid['Predictions'] = predictions

fig1=plt.figure(figsize=(13,8))

st.title('DECISION TREE :-')
plt.xlabel('Years')
plt.ylabel('Close Price USD ($)')
plt.plot(df2['Close'])
plt.plot(valid[['Close', 'Predictions']])
plt.legend(["Original", "Valid", 'Predicted'])
st.plotly_chart(fig1)
x, y, z = st.columns((3.4,3,2))
y.subheader('ORIGINAL vs PREDICTED')


st.markdown("""======================================================================================================================================================================""")

st.subheader('Final Prediction')
alpha3, gamma3 = st.columns(2)
alpha3.plotly_chart(fig4)
gamma3.plotly_chart(fig1)