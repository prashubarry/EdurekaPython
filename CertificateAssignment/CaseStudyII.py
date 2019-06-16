# -*- coding: utf-8 -*-
"""
Created on Tue Oct  2 19:01:20 2018

@author: Prashant Bhat
"""

"""
Loading the required Libraries
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
##############################################################################
"""
Loading the required datasets
"""
path = "G:\\extra things\\Knowledge\\practice\\Edureka\\CertificateAssignment\\574_proj_dataset_v3.0\\"
data = pd.read_csv(path+"Project_Data_1.csv",index_col=0,thousands=',')
data.index.names = ['country']
data.columns.names = ['years']
##############################################################################
"""
Exploring the data
"""
print(data.head(1))
print(data.describe())
print(data.columns)
print(data.dtypes)

#Check for columns with rows having NA- No such columns
null_columns = dict(data.isnull().any()[lambda x:x])
print(null_columns)

#############################################################################
"""
Applying Scaler to normalize the data from tons
"""

from sklearn.preprocessing import StandardScaler

scaler=StandardScaler()
scaler.fit(data)

scaled_data = scaler.transform(data)

"""
Now Apply PCA for 2 and then for 3 components
"""

from sklearn.decomposition import PCA

pca = PCA(n_components=2)

pca.fit(scaled_data)

x_pca = pca.transform(scaled_data)

print(scaled_data.shape)

print(x_pca.shape)

plt.figure(figsize=(9,6))
plt.scatter(scaled_data[:,0],scaled_data[:,1],c='r',alpha=0.5)
plt.scatter(x_pca[:,0],x_pca[:,1],c='b',alpha=0.5)
plt.xlabel('First Principal Component')
plt.ylabel('Second Principal Component')

print(pca.components_)
print(pca.explained_variance_ratio_)


#k=2
k=3
from sklearn.cluster import KMeans
from scipy.spatial.distance import cdist
kmeans = KMeans(n_clusters=k)
kmeans = kmeans.fit(x_pca)
labels = kmeans.predict(x_pca)
centroids = kmeans.cluster_centers_
print(kmeans.labels_)


print (type(kmeans.labels_))
unique, counts = np.unique(kmeans.labels_, return_counts=True)
print(dict(zip(unique, counts)))


colors = ['r', 'g', 'b', 'y', 'c', 'm']
fig, ax = plt.subplots()
for i in range(k):
        points = np.array([x_pca[j] for j in range(len(x_pca)) if labels[j] == i])
        ax.scatter(points[:, 0], points[:, 1], s=7, c=colors[i])
ax.scatter(centroids[:, 0], centroids[:, 1], marker='*', s=200, c='#050505')

#Selection of k with elbow method

K_range = range(1,10)
distortions = []

for i in K_range:
    kmeanModel = KMeans(n_clusters=i)
    kmeanModel.fit(x_pca)
    distortions.append(sum(np.min(cdist(x_pca,kmeanModel.cluster_centers_,'euclidean'),axis=1))/x_pca.shape[0])

fig1, ax1 = plt.subplots()
ax1.plot(K_range,distortions,"b*-")
plt.grid(True)
plt.ylim([0,45])
