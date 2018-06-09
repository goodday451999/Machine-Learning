import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# read dataset

dataset = pd.read_csv('Churn_Modelling - Churn_Modelling.csv')
X = dataset.iloc[:, 3:13].values
y = dataset.iloc[:,13].values

# encode categorical data

from sklearn.preprocessing import LabelEncoder, OneHotEncoder
le_X1 = LabelEncoder()
X[:, 2] = le_X1.fit_transform(X[:, 2])
le_X2 = LabelEncoder()
X[:, 2] = le_X2.fit_transform(X[:, 2])

ohe = OneHotEncoder.fit_transform(categorical_features=[1])
X = ohe.fit_transform(X).toarray()
X = X[:,1:]

# split the datset in train and test

from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2, random_state=0)

# fitting xgboost to the training set classifier
from xgboost import XGBClassifier
xc = XGBClassifier()
xc.fit(X_train, y_train)

# prediction

y_pred = xc.predict(X_test)

# confusion matrix

from sklearn.metrics import confusion_matrix
c_mat = confusion_matrix(y_test, y_pred)
