import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# read csv file

dataset = pd.read_csv('Position_Salaries.csv')
X = dataset.iloc[:, 1:2]
y = dataset.iloc[:, 2]

# linear regression classifier

from sklearn.linear_model import LinearRegression
linear_reg = LinearRegression()
linear_reg.fit(X, y)

# plot

plt.scatter(X, y, color='r')
plt.plot(X, linear_reg.predict(X), color='g')
plt.title('Linear Regression')
plt.xlabel('Position level')
plt.ylabel('Slary')
plt.show()
