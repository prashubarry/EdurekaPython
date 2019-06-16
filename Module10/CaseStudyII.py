# -*- coding: utf-8 -*-
"""
Created on Sun Sep 30 00:17:49 2018

@author: Prashant Bhat
"""
import numpy as np 
from PIL import Image
from sklearn.cluster import KMeans
from sklearn.metrics import pairwise_distances_argmin
from sklearn.utils import shuffle
from time import time
import matplotlib.pyplot as plt
plt.rcParams['figure.figsize'] = (5,5)
plt.style.use('ggplot')

data_img = Image.open("G:\\extra things\\Knowledge\\practice\\Edureka\\Module10\\dogs.jpeg")

n_colors = 3

dog = np.array(data_img,dtype=np.float64)/255
print(dog.shape)

# Load Image and transform to a 2D numpy array.
w, h, d = original_shape = tuple(dog.shape)
assert d == 3
image_array = np.reshape(dog, (w * h, d))
print(image_array.shape)
print("Fitting model on a small sub-sample of the data")
t0 = time()
image_array_sample = shuffle(image_array, random_state=0)[:1000]
kmeans = KMeans(n_clusters=n_colors, random_state=0).fit(image_array_sample)
print("done in %0.3fs." % (time() - t0))

# Get labels for all points
print("Predicting color indices on the full image (k-means)")
t0 = time()
labels = kmeans.predict(image_array)
print("done in %0.3fs." % (time() - t0))

codebook_random = shuffle(image_array, random_state=0)[:n_colors]
print("Predicting color indices on the full image (random)")
t0 = time()
labels_random = pairwise_distances_argmin(codebook_random,
                                          image_array,
                                          axis=0)
print("done in %0.3fs." % (time() - t0))

def recreate_image(codebook, labels, w, h):
    """Recreate the (compressed) image from the code book & labels"""
    d = codebook.shape[1]
    image = np.zeros((w, h, d))
    label_idx = 0
    for i in range(w):
        for j in range(h):
            image[i][j] = codebook[labels[label_idx]]
            label_idx += 1
    return image


# Display all results, alongside original image
plt.figure(1)
plt.clf()
plt.axis('off')
plt.title('Original image (50320 colors)')
plt.imshow(dog)

plt.figure(2)
plt.clf()
plt.axis('off')
plt.title('Quantized image (3 colors, K-Means)')
plt.imshow(recreate_image(kmeans.cluster_centers_, labels, w, h))

plt.figure(3)
plt.clf()
plt.axis('off')
plt.title('Quantized image (3 colors, Random)')
plt.imshow(recreate_image(codebook_random, labels_random, w, h))
plt.show()
