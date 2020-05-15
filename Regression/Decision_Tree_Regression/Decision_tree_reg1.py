import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

dataset = pd.read_csv("Position_salaries.csv")
X= dataset.iloc[:,1:2].values
Y= dataset.iloc[:,2].values

#Fitiing Decision Tree Reg 
from sklearn.tree import DecisionTreeRegressor
regressor= DecisionTreeRegressor(random_state= 0)
regressor.fit(X, Y)

#Predict
Y_pred= regressor.predict(6.5)

#Visualise
X_grid= np.arange(min(X), max(X), 0.01)
X_grid= X_grid.reshape((len(X_grid), 1))
plt.scatter(X, Y, color= 'Red')
plt.plot(X_grid, regressor.predict(X_grid), color= 'blue')
plt.title("DTR")
plt.xlabel("Level")
plt.ylabel("Salary")
plt.show()