# -*- coding: utf-8 -*-
"""
Created on Mon Jan  5 19:30:49 2026

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
svr_regressor=SVR(kernel='ploy',degree=4,gamma='auto',c=10)
svr_regressor.fit(x,y)

# workking with knn model

from sklearn.neighbors import KNeighborsRegressor
knn_reg_model=KNeighborsRegressor(n_neighbors=5,weights='distance',leaf_size=30)
knn_reg_model.fit(x,y)

knn_reg_pred=knn_reg_model.predict([[6.5]])
print(knn_reg_pred)

# decission tree
from sklearn.tree import DecisionTreeRegressor
dtr_reg_model=DecisionTreeRegressor(criterion='absolute_error',max_depth=10,splitter='random')
dtr_reg_model.fit(x,y)

dtr_reg_pred=dtr_reg_model.predict([[6.5]])
print(dtr_reg_pred)

# random forest
from sklearn.ensemble import RandomForestRegressor
rfr_reg_model=RandomForestRegressor(n_estimators=6,random_state=0)
rfr_reg_model.fit(x,y)

rfr_reg_pred=rfr_reg_model.predict([[6.5]])
print(rfr_reg_pred)

