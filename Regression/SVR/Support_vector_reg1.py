Support Vector regression
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

dataset= pd.read_csv("Position_Salaries.csv")
X= dataset.iloc[:,1:2].values
Y= dataset.iloc[:,2].values

#Feature Scaling
from sklearn.preprocessing import StandardScaler
sc_X= StandardScaler()
X= sc_X.fit_transform(X)
sc_Y= StandardScaler()
Y= sc_Y.fit_transform(Y.reshape(10,1))


#Fitting SVR to the dataset
from sklearn.svm import SVR
regressor= SVR(kernel='rbf')
regressor.fit(X,Y)

 #Predicting a new result
Y_pred = regressor.predict(sc_X.transform(np.array([[6.5]])))
Y_pred = sc_Y.inverse_transform(Y_pred)

#Visualising
plt.scatter(X, Y, color='Red')
plt.plot(X, regressor.predict(X), color='Blue')
plt.xlabel('Position Level')
plt.ylabel('Salaries')
plt.title("SVR")
plt.show() 
 