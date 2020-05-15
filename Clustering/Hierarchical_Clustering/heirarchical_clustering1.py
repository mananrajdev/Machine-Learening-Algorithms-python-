#Heirarchical Clustering
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

dataset= pd.read_csv("Mall_Customers.csv")
X= dataset.iloc[:,[3,4]].values

#using dendogram to find optimal no of clusters
import scipy.cluster.hierarchy as sch
dendogram= sch.dendrogram(sch.linkage(X, method= "ward"))
plt.title("Dendrogram")
plt.xlabel("Customers")
plt.ylabel("Eucleidian Distance")
plt.show()

#fitting HC
from sklearn.cluster import AgglomerativeClustering
hc= AgglomerativeClustering(n_clusters=5, affinity="euclidean", linkage='ward')
y_hc=hc.fit_predict(X)

#Visualisation
plt.scatter(X[y_hc==0,0],X[y_hc==0,1,], c='red', s=100, label="Cluster 1" )
plt.scatter(X[y_hc==1,0],X[y_hc==1,1,], c='blue', s=100, label="Cluster 2" )
plt.scatter(X[y_hc==2,0],X[y_hc==2,1,], c='green', s=100, label="Cluster 3" )
plt.scatter(X[y_hc==3,0],X[y_hc==3,1,], c='orange', s=100, label="Cluster 4" )
plt.scatter(X[y_hc==4,0],X[y_hc==4,1,], c='black', s=100, label="Cluster 5" )
plt.title("Cluster of customers")
plt.xlabel("Annual income")
plt.ylabel("Spending score")
plt.legend()
plt.show()


