# -*- coding: utf-8 -*-
"""
Created on Mon Jul  2 19:11:05 2018

@author: soanand

Problem Statement
Build the linear regression model using scikit learn in boston data 
to predict 'Price' based on other dependent variable.
"""

#Step 1. Import the required libraries
import numpy as np
import pandas as pd
import scipy.stats as stats
import matplotlib.pyplot as plt
import sklearn
from sklearn.datasets import load_boston

#Step 2. Read the dataset
boston = load_boston()
bos = pd.DataFrame(boston.data)

#Step 3. Data Preprocessing
bos.columns = boston.feature_names
bos['PRICE'] = boston.target

#Step 3. Divide independent and dependent dataset
X = bos.iloc[:,0:13]
y = bos.iloc[:,-1]

#Step 4. Divide the training and test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)

#Step 5. Create model
from sklearn.linear_model import LinearRegression
model = LinearRegression()
model.fit(X_train, y_train)

#Step6. Predicting the results for test dataset
y_pred = model.predict(X_test)

mse = sklearn.metrics.mean_squared_error(y_test, y_pred)
print(mse)

#Step 6. Plotting the graph for 
plt.scatter(y_test, y_pred)
plt.xlabel("Actual Price")
plt.ylabel("Predicted prices")
plt.title("Prices vs Predicted prices")


