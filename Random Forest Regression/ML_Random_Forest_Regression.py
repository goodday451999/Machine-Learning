import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# read the dataset

dataset = pd.read_csv('Position_Salaries.csv')
X = dataset.iloc[:, 1:2].values
y = dataset.iloc[:, 2].values

# random forest regression classifier

from sklearn.ensemble import RandomForestRegressor
rf_reg = RandomForestRegressor(n_estimators = 100, random_state = 0)
                            # n_estimators :: no of decision trees
rf_reg.fit(X, y)

# visualissing with higher resolution

X_grid = np.arange(min(X), max(X), 0.01)
X_grid = X_grid.reshape((len(X_grid), 1))
plt.scatter(X, y, color='r', s=100)
plt.plot(X, rf_reg.predict(X), color='g')
plt.plot(X_grid, rf_reg.predict(X_grid), color='blue')
plt.title('Random Forest Regression')
plt.xlabel('Position level')
plt.ylabel('Salary')                        
plt.show()    



                
