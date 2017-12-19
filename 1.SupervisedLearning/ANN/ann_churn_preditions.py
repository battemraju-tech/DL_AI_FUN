# Artificial Neural Network
# Installing Theano
# pip install --upgrade --no-deps git+git://github.com/Theano/Theano.git
# Installing Tensorflow
# pip install tensorflow
# Installing Keras
# pip install --upgrade keras
# Part 1 - Data Preprocessing
# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os

os.chdir('D:/Data/DataScience/DeepLearning/Artificial_Neural_Networks/Artificial_Neural_Networks/')

# Importing the dataset
dataset = pd.read_csv('Churn_Modelling.csv')
dataset.shape
dataset.info()
dataset[:1]

X = dataset.drop(['RowNumber','CustomerId','Surname'], axis=1, inplace=False)
X[:0]

X1 = pd.get_dummies(X, columns=['Geography', 'Gender'])
X1.shape
X1.info()
X1[:0]
y = dataset['Exited']

# Splitting the dataset into the Training set and Test set
from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X1, y, test_size = 0.20, random_state = 0)
X.shape
X.info()

# Feature Scaling: Needed ease of calulations
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

#Part2: Now Lets Make the ANN
import keras 
from keras.models import Sequential
from keras.layers import Dense#Randomly initilise the weights to small numbers close to 0(but not 0)
#Initializing ANN
classifier = Sequential()

# as first layer in a sequential model:
classifier.add(Dense(7, input_shape=(14,), activation='relu'))
# now the model will take as input arrays of shape (*, 16)
# and output arrays of shape (*, 32)

# after the first layer, you don't need to specify
# the size of the input anymore:

#ADDING Second hidden layer
classifier.add(Dense(7, activation='relu'))

#ADDING OUTPUT layer
classifier.add(Dense(1, activation='sigmoid'))

#COMPILING ANN
classifier.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

#Fitting the ANN to the training set
#classifier.fit(X_train, y_train, batch_size=10, epochs=50)
#total runs=30
#each run split data into 10 batches
classifier.fit(X_train, y_train, batch_size=10, epochs=30)

#Part-3: Making predictions and evaluating the model
#Predicting the test set results
y_pred = classifier.predict(X_test)

y_pred = y_pred>0.5

#Making confusion matrix
from sklearn.metrics import confusion_matrix
confMatrix = confusion_matrix(y_test,y_pred)

# Predicting a single new observation
"""Predict if the customer with the following informations will leave the bank:
Geography: France
Credit Score: 600
Gender: Male
Age: 40
Tenure: 3
Balance: 60000
Number of Products: 2
Has Credit Card: Yes
Is Active Member: Yes
Estimated Salary: 50000"""
#################Live Data Comes, now Lets test########################
# Importing the dataset
dataset = pd.read_csv('Churn_Modelling.csv')
dataset.shape
dataset.info()
dataset[:1]

X = dataset.drop(['RowNumber','CustomerId','Surname'], axis=1, inplace=False)
X[:0]

X1 = pd.get_dummies(X, columns=['Geography', 'Gender'])
X1.shape
X1.info()
X1[:0]
y = dataset['Exited']

# Splitting the dataset into the Training set and Test set
from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X1, y, test_size = 0.20, random_state = 0)
X.shape
X.info()

# Feature Scaling: Needed ease of calulations
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)



























