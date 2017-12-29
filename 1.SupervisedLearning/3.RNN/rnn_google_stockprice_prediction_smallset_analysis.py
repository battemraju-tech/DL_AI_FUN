#Recurrent Neural Network

#Part 1 - Data Preprocessing

#Importing the libraries
import numpy as np
import matplotlib as plt
import pandas as pd
import os

#Importing the training set
os.chdir('D:/Data/DataScience/DeepLearning/SuperDataScience/PART 3. RECURRENT NEURAL NETWORKS (RNN)/Recurrent_Neural_Networks/')
dataset_train_small = pd.read_csv('Google_Stock_Price_Train.csv')
training_set_small = dataset_train_small.iloc[0:10,1:2]

#Feature Scaling
from sklearn.preprocessing import MinMaxScaler
sc = MinMaxScaler(feature_range=(0,1))
training_set_scalar_small = sc.fit_transform(training_set_small)

#Creating a data structure with 60 timesteps and 1 output
X_train_small=[]
y_train_small=[]
for i in range(4,10):
    X_train_small.append(training_set_scalar_small[i-4:i,0])
    y_train_small.append(training_set_scalar_small[i,0])
    
X_train_small, y_train_small = np.array(X_train_small), np.array(y_train_small)

#Reshaping
X_train_small = np.reshape(X_train_small, (X_train_small.shape[0], X_train_small.shape[1],1))

#Part 2 - Building the RNN

#Part 3 - Making the predictions and visualization

dataset_test_small = pd.read_csv('Google_Stock_Price_Test.csv');
test_set_small = dataset_test_small.iloc[:,1:2]
total_set_small = pd.concat((training_set_small, test_set_small),axis=0)
total_set_small = total_set_small.values

#Get Predicted stock price of 2017 Jan
total_set_small2 = total_set_small[15:,]

