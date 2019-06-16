# -*- coding: utf-8 -*-
"""
Created on Sat Sep 29 22:18:39 2018

@author: Prashant Bhat
"""

"""
Importing required libraries
"""
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from copy import deepcopy
from scipy.spatial.distance import cdist
plt.rcParams['figure.figsize'] = (5,5)
plt.style.use('ggplot')
"""
Loading Data
"""
path = "G:\\extra things\\Knowledge\\practice\\Edureka\\Module10\\"
data = pd.read_csv(path+"driver-data.csv")
print(data.shape)
print(data.head)
print(data.describe())

"""
Check if the data contains null values
No null values
"""
null_columns = dict(data.isnull().any()[lambda x:x])
print (null_columns)

"""
First Method using the normal approach
Second Method using the kmeans 
"""

f1 = data['mean_dist_day'].values
f2 = data['mean_over_speed_perc'].values

X = np.array(list(zip(f1,f2)))
plt.scatter(f1,f2,c="black",s=50)

"""
Euclidean distance Calculator
"""
def dist(a,b,ax=1):
    return np.linalg.norm(a-b,axis=ax)
#Number of clusters
k = 4
#X coordinates of random centroids

C_x = np.random.randint(0,np.max(X)-20,size=k)

#Y Coordinates of random clusters

C_y = np.random.randint(0,np.max(X)-20,size=k)

C = np.array(list(zip(C_x,C_y)),dtype = np.float32)

print(C)

#Plotting with centroids
plt.scatter(f1,f2,c = '#050505',s=7)
plt.scatter(C_x,C_y,marker='*',s=200,c='g')
plt.xlabel('Distance Feature')
plt.ylabel('Speeding Percent')
plt.title('RAW SCatter Plot')

"""
Method1
"""
# To store the value of centroids when it updates
C_old = np.zeros(C.shape)
# Cluster Lables(0, 1, 2)
clusters = np.zeros(len(X))
# Error func. - Distance between new centroids and old centroids
error = dist(C, C_old, None)
# Loop will run till the error becomes zero
while error != 0:
    # Assigning each value to its closest cluster
    for i in range(len(X)):
        distances = dist(X[i], C)
        cluster = np.argmin(distances)
        clusters[i] = cluster
    # Storing the old centroid values
    C_old = deepcopy(C)
    # Finding the new centroids by taking the average value
    for i in range(k):
        points = [X[j] for j in range(len(X)) if clusters[j] == i]
        C[i] = np.mean(points, axis=0)
    error = dist(C, C_old, None)

colors = ['r', 'g', 'b', 'y', 'c', 'm']
fig, ax = plt.subplots()
for i in range(k):
        points = np.array([X[j] for j in range(len(X)) if clusters[j] == i])
        ax.scatter(points[:, 0], points[:, 1], s=7, c=colors[i])
ax.scatter(C[:, 0], C[:, 1], marker='*', s=200, c='#050505')
           
"""
Method2
"""

kmeans = KMeans(n_clusters=k)
kmeans = kmeans.fit(X)
labels = kmeans.predict(X)
centroids = kmeans.cluster_centers_


colors = ['r', 'g', 'b', 'y', 'c', 'm']
fig, ax = plt.subplots()
for i in range(k):
        points = np.array([X[j] for j in range(len(X)) if labels[j] == i])
        ax.scatter(points[:, 0], points[:, 1], s=7, c=colors[i])
ax.scatter(centroids[:, 0], centroids[:, 1], marker='*', s=200, c='#050505')

#Selection of k with elbow method

K_range = range(1,10)
distortions = []

for i in K_range:
    kmeanModel = KMeans(n_clusters=i)
    kmeanModel.fit(X)
    distortions.append(sum(np.min(cdist(X,kmeanModel.cluster_centers_,'euclidean'),axis=1))/X.shape[0])

fig1, ax1 = plt.subplots()
ax1.plot(K_range,distortions,"b*-")
plt.grid(True)
plt.ylim([0,45])