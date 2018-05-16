import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# read the dataset

dataset = pd.read_csv('Positon_Salaries.csv')
X = dataset.iloc[:, 1:2].values
y = dataset.iloc[:, 2].values

# simple linear regression classifier

from sklearn.linear_model import LinearRegression
linear_reg1 = LinearRegression()
linear_reg1.fit(X, y)

# ploynomial regression

from sklearn.preprocessing import PolynomialFeatures
poly_reg = PolynomialFeatures(degree=4)
X_Poly = poly_reg.fit_transform(X)

linear_reg2 = LinearRegression()
linear_reg2.fit(X_Poly, y)

# plot simple linear regression

plt.scatter(X, y, color='r')
plt.plot(X, linear_reg1.predict(X), color='g')
plt.show()

# plot polynomial regression

plt.scatter(X, y, color='blue')
plt.plot(X, linear_reg2.predict(X_Poly), color='m')
plt.show()
