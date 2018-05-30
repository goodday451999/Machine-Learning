import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# dataset read

dataset = pd.read_csv('Mall_Customers - Mall_Customers.csv')
X = dataset.iloc[:, [3,4]].values

# elbow method : to find the optimal number of clusters

from sklearn.cluster import KMeans
        # within cluster sum of squares
wcss=[]
        # for 10 different clusters
for i in range(1, 11):
    # initialization of the centroids-- there is an optimization values ==
    # = cost function and the lowest cost will be taken
    kmeans = KMeans(n_clusters=i, init='k-means++', random_state=0)
    kmeans.fit(X)
    wcss.append(kmeans.inertia_)
    # to give wcss value of each and every clusters away as the for loop goes

# plot Elbow

plt.plot(range(1, 11), wcss)
plt.title('The Elbow Method')
plt.xlabel('Number of Clusters')
    # WCSS Value is the Euclidian Distance between the scattered points
plt.ylabel('WCSS Value') 
plt.show()

    # In the output graph we can find a "human's elbow" kind structure
    # and by determining the minimum value of the elbow we can perdict the number of clusters

# fitting KMeans to the dataset

    # for this problem number of clusters will be 5
kmeans = KMeans(n_clusters=5, init='k-means++', random_state=0)
y_kmeans = kmeans.fit_predict(X)
    # only fit will provide the extream centroids only

# visualize the clusters

    # we already determined that we have to form 5 clusters
    # for each clusters 0th feature is belonging to the y_kmeans==0 i.e, 0th cluster
    # similarlt 1st feature is belonging to the y_kmeans==0 i.e, 0th cluster
plt.scatter(X[y_kmeans==0,0], X[y_kmeans==0,1], s=100, color='r', label='Cluster1')
plt.scatter(X[y_kmeans==1,0], X[y_kmeans==1,1], s=100, color='b', label='Cluster2')
plt.scatter(X[y_kmeans==2,0], X[y_kmeans==2,1], s=100, color='c', label='Cluster3')
plt.scatter(X[y_kmeans==3,0], X[y_kmeans==3,1], s=100, color='m', label='Cluster4')
plt.scatter(X[y_kmeans==4,0], X[y_kmeans==4,1], s=100, color='g', label='Cluster5')

# plot centroids
   
plt.scatter(kmeans.cluster_centers_[:,0], kmeans.cluster_centers_[:,1], s=200, color='y', label='Centroids')
plt.title('KMean Clusters of Customers')
plt.xlabel('Annual Income(K$)')
plt.ylabel('Spending Score(1-100)')
plt.show()
