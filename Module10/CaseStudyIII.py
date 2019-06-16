# -*- coding: utf-8 -*-
"""
Created on Sun Sep 30 17:17:53 2018

@author: Prashant Bhat
"""

"""
Import the required libraries
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
plt.rcParams['figure.figsize'] = (5,5)
plt.style.use('ggplot')

"""
Loading the dataset 
"""

"""
1.Load the file “zoo.data” and look at the info and first five rows. 
The first column denotes the animal name and the last one specifies a high-level class 
for the corresponding animal.
"""
path = "G:\\extra things\\Knowledge\\practice\\Edureka\\Module10\\"
data = pd.read_csv(path+"zoo.csv")

print(data.head(5))
print(data.describe())
print(data.info())
"""
Check for null rows columns
nothing
"""

null_columns = dict(data.isnull().any()[lambda x:x])
print (null_columns)

data.head(1)

"""
Find out the unique number of high level class
"""

data.groupby('class_type')['class_type'].count()

data.groupby('animal_name').groups.keys()

"""
3. Use the 16-intermediate feature and perform an agglomerative clustering.
"""

X = data.iloc[:,[0,17]]
X.head(1)
y = data['animal_name']
from sklearn.preprocessing import LabelEncoder
label_encoder = LabelEncoder()
y = label_encoder.fit_transform(y)
X['animal_name1'] = y
X.head(1)

X1 = X.iloc[:,[1,2]]
X1.head(1)
#Using Dendogram to find the optimal number of clusters

import scipy.cluster.hierarchy as sch

dendrogram = sch.dendrogram(sch.linkage(X1,method='ward'))
plt.title('Dendogram')
plt.xlabel('Animals')
plt.ylabel('Eucledian Distance')
plt.show() 
#Import module for hierarchial clustering
from sklearn.cluster import AgglomerativeClustering

hc = AgglomerativeClustering(n_clusters=16,affinity='euclidean',linkage='ward')
y_hc = hc.fit_predict(X1)


"""
4. Compute the mean squared error by comparing the actual class and predicted high level class.
"""
from sklearn.metrics import mean_squared_error
print(mean_squared_error(X1['animal_name1'],y_hc))

