#Random Forest 
#Decision Tree Reg
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

dataset= pd.read_csv("Social_Network_Ads.csv")
X= dataset.iloc[:,[2,3]].values
Y= dataset.iloc[:,4].values

from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test= train_test_split(X, Y, test_size=0.25, random_state= 0)

from sklearn.preprocessing import StandardScaler
sc_X= StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test= sc_X.transform(X_test)

#Fitting Random Forest Classification
from sklearn.ensemble import RandomForestClassifier
classifier= RandomForestClassifier(n_estimators=10, criterion="entropy", random_state=0)
classifier.fit(X_train, Y_train)

#Predict
y_pred= classifier.predict(X_test)

#Confusion Matrix
from sklearn.metrics import confusion_matrix
cm= confusion_matrix(Y_test, y_pred)

#Visualise training set
from matplotlib.colors import ListedColormap
x_set, y_set= X_train, Y_train
X1, X2= np.meshgrid(np.arange(start= x_set[:,0].min()-1, stop= x_set[:,0].max()+1, step= 0.01), np.arange(start= x_set[:,1].min()-1, stop= x_set[:,1].max()+1, step= 0.01))
plt.contourf(X1, X2, classifier.predict(np.array([X1.ravel(), X2.ravel()]).T).reshape(X1.shape), aplha= 0.75, cmap= ListedColormap(("red", "green")))
plt.xlim(X1.min(), X1.max())
plt.ylim(X2.min(), X2.max())

for i, j in enumerate(np.unique(y_set)):
    plt.scatter(x_set[y_set==j, 0], x_set[y_set==j, 1], c= ListedColormap(('red', 'green'))(i), label=j)
    
plt.title("Decision Tree")
plt.xlabel("Age")
plt.ylabel("Salary")
plt.legend()
plt.show()

#Visualise test set
from matplotlib.colors import ListedColormap
x_set, y_set= X_test, Y_test
X1, X2= np.meshgrid(np.arange(start= x_set[:,0].min()-1, stop= x_set[:,0].max()+1, step= 0.01), np.arange(start= x_set[:,1].min()-1, stop= x_set[:,1].max()+1, step= 0.01))
plt.contourf(X1, X2, classifier.predict(np.array([X1.ravel(), X2.ravel()]).T).reshape(X1.shape), aplha= 0.75, cmap= ListedColormap(("red", "green")))
plt.xlim(X1.min(), X1.max())
plt.ylim(X2.min(), X2.max())

for i, j in enumerate(np.unique(y_set)):
    plt.scatter(x_set[y_set==j, 0], x_set[y_set==j, 1], c= ListedColormap(('red', 'green'))(i), label=j)
    
plt.title("Decision Tree")
plt.xlabel("Age")
plt.ylabel("Salary")
plt.legend()
plt.show()
