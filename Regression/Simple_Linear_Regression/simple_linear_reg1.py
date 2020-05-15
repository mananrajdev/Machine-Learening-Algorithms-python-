#Import Library
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

#Import dataset
dataset= pd.read_csv("Salary_Data.csv")
X= dataset.iloc[:,:-1].values
Y= dataset.iloc[:,1].values

#Split dataset
from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test= train_test_split(X, Y, test_size=1/3, random_state=0)

#Fitting SLR to training set
from sklearn.linear_model import LinearRegression
regressor= LinearRegression()
regressor.fit(X_train, Y_train)

#Predicting the test set results
Y_pred= regressor.predict(X_test)

#Visualising the training set
plt.scatter(X_train, Y_train, color='red')
plt.plot(X_train, regressor.predict(X_train), color='blue')
plt.title("Salary VS Experience (Training set)")
plt.xlabel("Experience")
plt.ylabel("Salary")
plt.show()
