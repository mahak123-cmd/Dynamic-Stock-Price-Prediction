import numpy as np
import pandas as pd
import yfinance as yf
from keras.models import load_model
import streamlit as st
import matplotlib.pyplot as plt

from datetime import date
from prophet import Prophet
from plotly import graph_objs as go
from prophet.plot import plot_plotly

st.set_page_config(layout="wide")

images0,image1,images1 = st.columns((2.5,6,2))
image1.image('https://miro.medium.com/max/620/0*dunTLlei47QWR7NR.gif')
alpha, beta, gamma = st.columns((4,5,3))
beta.title('Stock Price Prediction')

alpha1, gamma1 = st.columns(2)

alpha1.markdown('Hello and Welcome to Stock Price Predictor. This WebApp is designed to predict the price of select stocks. To get started, Select a stock from the given list and select the starting date and ending date for the same to view its information.')


stock = gamma1.markdown('Stock price prediction is a very researched entity in the modern day world. It helps companies to raise capital, it helps people generate passive income, stock markets represent the state of the economy of the country and it is widely used soutce for people to invest money in companies with high growth potential')

st.markdown("""=======================================================================================================================================================""")

stock = st.text_input('Enter Stock Symbol', 'GOOG')
start = '2012-01-01'
end = '2023-12-01'

df = yf.download(stock, start, end)

rhs, rh, rhs1 = st.columns((2.5,3,2))
rh.markdown("""### Let's have a look at some raw data""")
rds, rd, rds1 = st.columns((2.5,5,1))
rd.write(df)

df['Open - Close']=df['Open'] - df['Close']
df['High - Low'] = df['High'] - df['Low']

X = df[['Open - Close', 'High - Low']]


Y= np.where(df['Close'].shift(-1)>df['Close'],1,-1)

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

st.markdown("""=======================================================================================================================================================""")

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
rhs.write('Train_data Accuracy: %.2f' %accuracy_train)
rh.write('Test_data Accuracy: %.2f' %accuracy_test)

predictions_classification = model.predict(X_test)

actual_predicted_data = pd.DataFrame({'Actual Class':y_test, 'Predicted Class':predictions_classification})
actual_predicted_data.head(10)
x, y, z = st.columns((2.7,3,2))
y.markdown("""### KNN CLASSIFIER""")
p, q, r = st.columns((2.5,3,2))
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


#rmse
rms=np.sqrt(np.mean(np.power((np.array(y_test)-np.array(predictions)),2)))
st.write('RMSE : ',rms)

valid = pd.DataFrame({'Actual Price':y_test_reg, 'Predicted Price value':predictions})
d, e, f = st.columns((2.5,3,2))
e.markdown("""### KNN REGRESSION""")
a, b, c = st.columns((2.5,5,1))
b.write(valid)


