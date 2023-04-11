#building an AI model that detects a rock or mine

#importing files
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

#loading the dataset to a pandas Dataframe
sonar_data = pd.read_csv('/content/Copy of sonar data.csv', header = None)

#number of columns and rows
sonar_data.shape

#208 data points 61 features

#describe --> statistical measures of data
sonar_data.describe()

sonar_data[60].value_counts()
sonar_data.groupby(60).mean()

#seperate data and labels
X = sonar_data.drop(columns=60, axis=1)
Y = sonar_data[60]
print (X)
print (Y)
X_train, X_test, Y_train, Y_test = train_test_split(X,Y, test_size=0.1, stratify=Y, random_state=1)

print(X.shape, X_train.shape, X_test.shape)

#Model Training
model = LogisticRegression()
model.fit(X_train, Y_train)

#Model Evaluation

#accuracy on training data
X_train_prediction = model.predict(X_train)
training_data_accuracy = accuracy_score(X_train_prediction, Y_train)
print('Accuracy on training data : ', training_data_accuracy)

#accuracy on test data
X_test_prediction = model.predict(X_test)
test_data_accuracy = accuracy_score(X_test_prediction, Y_test)
print('Accuracy on test data : ', test_data_accuracy)

#making a predictive solution

#changing the input data to numpy array
input_data = (0.0094,0.0611,0.1136,0.1203,0.0403,0.1227,0.2495,0.4566,0.6587,0.5079,0.3350,0.0834,0.3004,0.3957,0.3769,0.3828,0.1247,0.1363,0.2678,0.9188,0.9779,0.3236,0.1944,0.1874,0.0885,0.3443,0.2953,0.5908,0.4564,0.7334,0.1969,0.2790,0.6212,0.8681,0.8621,0.9380,0.8327,0.9480,0.6721,0.4436,0.5163,0.3809,0.1557,0.1449,0.2662,0.1806,0.1699,0.2559,0.1129,0.0201,0.0480,0.0234,0.0175,0.0352,0.0158,0.0326,0.0201,0.0168,0.0245,0.0154)
input_data_as_numpy_array = np.asarray(input_data)

#reshape the np array as we are predicting for one instance
input_data_reshaped = input_data_as_numpy_array.reshape(1,-1)

prediction = model.predict(input_data_reshaped)
print(prediction)
