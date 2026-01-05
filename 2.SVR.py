# -*- coding: utf-8 -*-
"""
Created on Mon Jan  5 19:11:55 2026

@author: Prashanth
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
# importing dataset
dataset=pd.read_csv(r"C:\Users\Prashanth\Desktop\Naresh_it\MachineLearning\Data\emp_sal.csv") 
x=dataset.iloc[:,1:2].values
y=dataset.iloc[:,2].values

# fitting svr to the dataset

from sklearn.svm import SVR
regressor=SVR()
regressor.fit(x,y)

y_pred=regressor.predict([[6.5]])


# visualizing the  svr plot

plt.scatter(x,y,color='red')
plt.plot(x,regressor.predict(x),color='blue')
plt.title('Truth or bluff(svr')
plt.xlabel('position level')
plt.ylabel('salary')
plt.show()

# Visualising the SVR results (for higher resolution and smoother curve)
x_grid = np.arange(min(x), max(x), 0.01) 
x_grid = x_grid.reshape((len(x_grid), 1))
plt.scatter(x, y, color = 'red')
plt.plot(x_grid, regressor.predict(x_grid), color = 'blue')
plt.title('Truth or Bluff (SVR)')
plt.xlabel('Position level')
plt.ylabel('Salary')
plt.show()

