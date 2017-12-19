#Evaluating the ANN
#Evaluating(Cross Validation(cv), GridSearch CV)
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import cross_val_score
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


