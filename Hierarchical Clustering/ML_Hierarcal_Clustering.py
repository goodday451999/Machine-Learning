import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# read dataset

dataset = pd.read_csv('Mall_Customers - Mall_Customers.csv')
X = dataset.iloc[:, [3,4]].values

# dendogram

import scipy.cluster.hierarchy as sch
dg = sch.dendrogram(sch.linkage(X, method='ward'))
    # method 'ward' create the hierarchy
    # dendogram is the hierarchy of variables present in X
plt.title('DENDROGRAM')
plt.xlabel('Customers')
plt.ylabel('Eucledian Distance')
    # Euclidian distance between all (3,4) for each rows in X
plt.show()

    # in the outpur figure we can determine the numbers of clusters
    # Method I  : By observing the different colours
    #--- here it is 3
    # Method II : Longest vertical lines through which no horizontal line passes
    #--- here it is 5

# fitting Hierarchical to the dataset

from sklearn.cluster import AgglomerativeClustering
h_c = AgglomerativeClustering(n_clusters=5, affinity="euclidean", linkage='ward')
y_h_c = h_c.fit_predict(X)

# visualization of the clusters

plt.scatter(X[y_h_c==0, 0], X[y_h_c==0, 1], s=100, c='r', label='Cluster1')
plt.scatter(X[y_h_c==1, 0], X[y_h_c==1, 1], s=100, c='b', label='Cluster2')
plt.scatter(X[y_h_c==2, 0], X[y_h_c==2, 1], s=100, c='c', label='Cluster3')
plt.scatter(X[y_h_c==3, 0], X[y_h_c==3, 1], s=100, c='m', label='Cluster4')
plt.scatter(X[y_h_c==4, 0], X[y_h_c==4, 1], s=100, c='g', label='Cluster5')

# plot centroids
     # not supported
#plt.scatter(h_c.cluster_centers_[:, 0], h_c.cluster_centers_[:, 1], s=200, c='y', label='Centroid')
plt.title('Hierarchial Clustering')
plt.xlabel('Annual Income(K$)')
plt.ylabel('Spending Score(1-100)')
plt.show()
