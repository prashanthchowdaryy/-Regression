# -*- coding: utf-8 -*-
"""
Created on Mon Jan  5 19:47:25 2026

@author: Prashanth
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
# importing dataset
dataset=pd.read_csv(r"C:\Users\Prashanth\Desktop\Naresh_it\MachineLearning\Data\emp_sal.csv") 
x=dataset.iloc[:,1:2].values
y=dataset.iloc[:,2].values

from sklearn.tree import DecisionTreeRegressor
regressor=DecisionTreeRegressor(criterion='friedman_mse',splitter='random')
regressor.fit(x,y)

from sklearn.ensemble import RandomForestRegressor
reg=RandomForestRegressor(n_estimators=300,random_state=0)
reg.fit(x,y)

# predecting a new result
y_pred=reg.predict([[6.5]])


plt.scatter(x, y, color = 'red')
plt.plot(x,regressor.predict(x), color = 'blue')
plt.title('Truth or bluff (Decision tree Regression)')
plt.xlabel('Position Level')
plt.ylabel('salary')
plt.show()

# Visualising the Decision Tree Regression results (higher resolution)
x_grid = np.arange(min(x), max(x), 0.01)
x_grid = x_grid.reshape((len(x_grid), 1))
plt.scatter(x, y, color = 'red')
plt.plot(x_grid, regressor.predict(x_grid), color = 'blue')
plt.title('Truth or Bluff (Decision Tree Regression)')
plt.xlabel('Position level')
plt.ylabel('Salary')
plt.show()

