#Polynomial Regression
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

dataset= pd.read_csv("Position_Salaries.csv")
X= dataset.iloc[:,1:2].values
Y= dataset.iloc[:,2].values


"""from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=0)"""

#Linear regressiom
from sklearn.linear_model import LinearRegression
linreg= LinearRegression()
linreg.fit(X, Y)
  
#Polynomial Regressiom
from sklearn.preprocessing import PolynomialFeatures
poly_reg= PolynomialFeatures(degree= 4)
X_poly= poly_reg.fit_transform(X)

linreg2=LinearRegression()
linreg2.fit(X_poly, Y)

#Visualising the Linear reg result
plt.scatter(X, Y, color='Red')
plt.plot(X, linreg.predict(X), color='Blue')
plt.title("Linear Regression")
plt.xlabel("Position level")
plt.ylabel("Salary")
plt.show()

#Visualising Polynomial Regression
X_grid= np.arange(min(X), max(X), 0.1)
X_grid= X_grid.reshape((len(X_grid)),1)
plt.scatter(X, Y, color= "Red")
plt.plot(X_grid, linreg2.predict(poly_reg.fit_transform(X_grid)), color="Blue")
plt.title("Polynomial Regression")
plt.xlabel("Position level")
plt.ylabel("Salary")
plt.show()

#Predict the new result with Linear Reg
linreg.predict(6.5)

#Predict the new result with Poly Reg
linreg2.predict(poly_reg.fit_transform(6.5))