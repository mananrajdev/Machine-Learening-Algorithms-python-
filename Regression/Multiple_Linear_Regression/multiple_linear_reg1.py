import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

dataset=pd.read_csv('50_Startups.csv')
X= dataset.iloc[:,:-1].values
Y= dataset.iloc[:,4].values

from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labenc= LabelEncoder()
X[:,3]=labenc.fit_transform(X[:,3])
ohe= OneHotEncoder(categorical_features= [3])
X= ohe.fit_transform(X).toarray()

#Avoiding dummy variable trap
X=X[:,1:]
    

from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X,Y, test_size=0.2, random_state=0) 

#Fitting Multiple Linear Regression to the training set
from sklearn.linear_model import LinearRegression
regressor= LinearRegression()
regressor.fit(X_train, Y_train)

#Predicting the test set
Y_pred= regressor.predict(X_test)

#Backward Elimination
import statsmodels.formula.api as sm
X= np.append(arr=np.ones((50,1)).astype(int), values=X, axis=1)
X_opt= X[:,[0,1,2,3,4,5]]
regressor_OLS= sm.OLS(endog=Y, exog=X_opt).fit()
regressor_OLS.summary()

X_opt=X[:,[0,1,3,4,5]]
regressor_OLS=sm.OLS(endog=Y, exog= X_opt).fit()
regressor_OLS.summary()

X_opt=X[:,[0,3,4,5]]
regressor_OLS=sm.OLS(endog=Y, exog=X_opt).fit()
regressor_OLS.summary()

X_opt=X[:,[0,3,5]]
regressor_OLS=sm.OLS(endog=Y, exog=X_opt).fit()
regressor_OLS.summary()

X_opt=X[:,[0,3]]
regressor_OLS=sm.OLS(endog= Y, exog=X_opt).fit()
regressor_OLS.summary()