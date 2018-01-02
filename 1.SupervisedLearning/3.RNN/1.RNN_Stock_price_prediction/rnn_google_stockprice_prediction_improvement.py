#Recurrent Neural Network

#Part 1 - Data Preprocessing

#Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
from sklearn import model_selection, metrics
from keras.wrappers.scikit_learn import KerasRegressor
import math

#Importing the training set
os.chdir('D:/Data/DataScience/DeepLearning/SuperDataScience/PART 3. RECURRENT NEURAL NETWORKS (RNN)/Recurrent_Neural_Networks/')
dataset_train = pd.read_csv('Google_Stock_Price_Train.csv')
training_set = dataset_train.iloc[:,1:2].values

#Feature Scaling
from sklearn.preprocessing import MinMaxScaler
sc = MinMaxScaler(feature_range=(0,1))
training_set_scalar = sc.fit_transform(training_set)

#Creating a data structure with 60 timesteps and 1 output
X_train=[]
y_train=[]
for i in range(60,1258):
    X_train.append(training_set_scalar[i-60:i,0])
    y_train.append(training_set_scalar[i,0])
    
X_train, y_train = np.array(X_train), np.array(y_train)

#Reshaping
X_train.shape
X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))

#Part 2 - Building the RNN
#Importing keras libraries and packages
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Dropout

#Initializing the RNN
regressor = Sequential()

#Adding the first LSTM layer and Some Dropout regularisation
#units=50, #no of neurons in network
#dropout=0.2, #20% of neurons can ignored in network
#return_sequences =True, means return 
#input_shape=(X_train.shape[1], 1)#60,1. 

#regressor.add(LSTM(units=50, dropout=0.2,return_sequences =True,input_shape=(X_train.shape[1], 1)))

#Adding the second LSTM layer and Some Dropout regularisation
#regressor.add(LSTM(units=50, dropout=0.2, return_sequences =True))

#Adding the third LSTM layer and Some Dropout regularisation
#regressor.add(LSTM(units=50, dropout=0.2, return_sequences =True))

#Adding the fourth LSTM layer and Some Dropout regularisation
#regressor.add(LSTM(units=50, dropout=0.2)) # default return_sequence =False

#Adding output layer
#regressor.add(Dense(units=1))

#Compiling the RNN
#regressor.compile(optimizer='adam', loss='mean_squared_error')
# why not loss='category_crossentropy'??

#Training RNN Model with training set

#regressor.fit(X_train, y_train, epochs=100, batch_size=32)


def build_regressor():
    regressor.add(LSTM(units=50, dropout=0.2,return_sequences =True, input_shape=(X_train.shape[1],1)))
    regressor.add(LSTM(units=50, dropout=0.2, return_sequences =True))
    regressor.add(LSTM(units=50, dropout=0.2, return_sequences =True))
    regressor.add(LSTM(units=50, dropout=0.2))
    regressor.add(Dense(units=1))
    regressor.compile(optimizer='adam', loss='mean_squared_error')
    return regressor

rnn_estimator = KerasRegressor(build_fn=build_regressor,batch_size=32,nb_epoch=100)

def rmse(y_orig, y_pred):
    return math.sqrt(metrics.mean_squared_error(y_orig,y_pred))

###Grid Search CV
rnn_grid = {}
grid_rnn_estimator = model_selection.GridSearchCV(rnn_estimator,rnn_grid,scoring=metrics.make_scorer(rmse))
grid_rnn_estimator.fit(X_train, y_train)
print(grid_rnn_estimator.grid_scores_)
print(grid_rnn_estimator.best_params_)
print(grid_rnn_estimator.best_score_)
print(grid_rnn_estimator.score(X_train, y_train))
estimator = grid_rnn_estimator.best_estimator_



#Part 3 - Making the predictions and visualization

#Get Prediction From RNN trained model, and Visualize actual and predicted stock prices.
dataset_test = pd.read_csv('Google_Stock_Price_Test.csv');
real_stock_price = dataset_test.iloc[:, 1:2].values
dataset_total = pd.concat((dataset_train['Open'], dataset_test['Open']), axis = 0)
inputs = dataset_total[len(dataset_total) - len(dataset_test) - 60:].values
inputs = inputs.reshape(-1,1)
inputs = sc.transform(inputs)
X_test = []
for i in range(60, 80):
    X_test.append(inputs[i-60:i, 0])
X_test = np.array(X_test)
X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))
predicted_stock_price = regressor.predict(X_test)
predicted_stock_price = sc.inverse_transform(predicted_stock_price)

#Visualize actual stocks with predicted stocks
plt.plot(real_stock_price, color='red', label='Real Google Stock Price')
plt.plot(predicted_stock_price, color='blue', label='Predicted Google Stock Price')
plt.title('Google Stock Price Prdiction')
plt.xlabel('Time')
plt.ylabel('Google Stock Price')
plt.legend()
plt.show()

