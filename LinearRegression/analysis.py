# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
ourData = pd.read_csv('Data.csv')
X = ourData.iloc[:, :-1].values
Y = ourData.iloc[:, -1].values

from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.2, 
                                                    random_state = 0)

# Feature scaling
from sklearn.preprocessing import StandardScaler 
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)

sc_Y = StandardScaler()
Y_train = sc_Y.fit_transform(Y_train)
Y_test = sc_Y.transform(Y_test)

# Fitting the model to our training set
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X=X_train, y=Y_train)

# Predict the values
ourPred = regressor.predict(X=X_test)

# Visualising the training set results
plt.scatter(x=X_train, y=Y_train, color = 'red')
plt.plot(X_train, regressor.predict(X_train), color = 'blue')
plt.title("Salary Vs Experience (Training set)")
plt.xlabel("Experience")
plt.ylabel("Salary")
plt.show()
plt.savefig("Salary Vs Experience (Training set).png")

# Visualising the testing set results
plt.scatter(x=X_test, y=ourPred, color = 'red')
plt.scatter(x=X_test, y=Y_test, color = 'green')
plt.plot(X_train, regressor.predict(X_train), color = 'blue')
plt.title("Salary Vs Experience (Testing set)")
plt.xlabel("Experience")
plt.ylabel("Salary")
plt.show()
plt.savefig("Salary Vs Experience (Testing set).png")
