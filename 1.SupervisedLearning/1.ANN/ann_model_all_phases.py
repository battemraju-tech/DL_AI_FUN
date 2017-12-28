# Artificial Neural Network
# Installing Theano
# pip install --upgrade --no-deps git+git://github.com/Theano/Theano.git
# Installing Tensorflow
# pip install tensorflow
# Installing Keras
# pip install --upgrade keras
# Part 1 - Data Preprocessing
# Importing the libraries
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
X_train.shape
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

#Part3
########################Live Data Testing##########################
# Predicting a single new observation - Which comes from live
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
dataset_live = pd.read_csv('Churn_Modelling_Live.csv')
dataset_live.shape
dataset_live.info()
dataset_live[:1]

X_live = dataset_live.drop(['RowNumber','CustomerId','Surname'], axis=1, inplace=False)
X_live[:0]

#Missing values handling
float_columns = ['CreditScore','Age','Tenure','Balance','NumOfProducts','HasCrCard','IsActiveMember','EstimatedSalary']
X_live[float_columns] = X_live[float_columns].fillna(0.0)
gender_column = ['Gender']
X_live[gender_column] = X_live[gender_column].fillna('Male')
target_column = ['Exited']
X_live[target_column] = X_live[target_column].fillna('0')
#One hot encoding
X1_live = pd.get_dummies(X_live, columns=['Geography', 'Gender'])
X1_live.shape
X1_live.info()
X1_live[:0]
y_live = X_live['Exited']

# Splitting the dataset into the Training set and Test set
# Splitting not required
# Feature Scaling: Needed ease of calulations
X_live_trans = sc.transform(X1_live)

#Predicting Live data
y_pred_live = classifier.predict(X_live_trans)
y_pred_live = y_pred_live>0.5
#output got False. It means customer will NOT Exit the bank. :) :) 




#############Evaluation/Accuracy Calculation for Given Data Using Cross Validation(KFold)##########################
#Part4: Evaluating(accuracy, cv, gridSearchSv), Improving(Regularization) and Tuning the ANN

#Evaluating the ANN
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import cross_val_score,GridSearchCV
from keras.models import Sequential
from keras.layers import Dense
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
X_train.shape
X.info()

# Feature Scaling: Needed ease of calulations
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

def build_classifier():
    ann_estimator = Sequential()
    ann_estimator.add(Dense(7, input_shape=(14,), activation='relu'))
    ann_estimator.add(Dense(7, activation='relu'))
    ann_estimator.add(Dense(1, activation='sigmoid'))
    ann_estimator.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return ann_estimator

ann_estimator = KerasClassifier(build_fn=build_classifier,batch_size=10,nb_epoch=30)

cv_scores = cross_val_score(ann_estimator, X=X_train,y=y_train,cv=10,n_jobs=1)

#Improving the ANN
#Improving(Regularization) -if overfit
from keras.layers import Dropout
def build_classifier():
    ann_estimator = Sequential()
    ann_estimator.add(Dense(7, input_shape=(14,), activation='relu'))
    ann_estimator.add(Dropout(p=0.1))
    ann_estimator.add(Dense(7, activation='relu'))
    ann_estimator.add(Dropout(p=0.1))
    ann_estimator.add(Dense(1, activation='sigmoid'))
    ann_estimator.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return ann_estimator

ann_estimator = KerasClassifier(build_fn=build_classifier,batch_size=10,nb_epoch=20)

cv_scores = cross_val_score(ann_estimator, X=X_train,y=y_train,cv=10,n_jobs=1)

cv_scores.mean()
variance = cv_scores.std()

#Hyperparameter Tuning

def build_classifier(optimizer):
    ann_estimator = Sequential()
    ann_estimator.add(Dense(7, input_shape=(14,), activation='relu'))
    ann_estimator.add(Dropout(p=0.1))
    ann_estimator.add(Dense(7, activation='relu'))
    ann_estimator.add(Dropout(p=0.1))
    ann_estimator.add(Dense(1, activation='sigmoid'))
    ann_estimator.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy'])
    return ann_estimator

ann_estimator = KerasClassifier(build_fn=build_classifier)

ann_grid = {'batch_size':[3,4],
        'nb_epoch':[20,25],
        'optimizer':['adam','rmsprop']}

grid_ann_estimator = GridSearchCV(ann_estimator,ann_grid,cv=10,n_jobs=1)

grid_ann_estimator.fit(X_train, y_train)

print(grid_ann_estimator.grid_scores_)
print(grid_ann_estimator.best_params_)
print(grid_ann_estimator.best_score_)



























