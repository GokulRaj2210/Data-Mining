import numpy as np
from matplotlib import pyplot as plt
from sklearn.datasets.samples_generator import make_blobs

#generating 100 data set in n sample
#your standard devation\sigma will be in cluster_std
#center or mean points then change center_box (any 2 dimensional data set change center_box)
X_1, y_1 = make_blobs(n_samples=100, center_box=(5,5), cluster_std=4, random_state=0)
X_2, y_2 = make_blobs(n_samples=100, center_box=(-5,5), cluster_std=4, random_state=0)
X_3, y_3 = make_blobs(n_samples=100, center_box=(-5,-5), cluster_std=4, random_state=0)
X_4, y_4 = make_blobs(n_samples=100, center_box=(5,-5), cluster_std=4, random_state=0)

#change here as well
x_1, y_1 = np.random.multivariate_normal([5,5],[[4,0],[0,4]], 100).T
x_2, y_2 = np.random.multivariate_normal([-5,5],[[4,0],[0,4]], 100).T
x_3, y_3 = np.random.multivariate_normal([-5,-5],[[4,0],[0,4]], 100).T
x_4, y_4 = np.random.multivariate_normal([5,-5],[[4,0],[0,4]], 100).T

import pandas as pd
plt.plot(x_1,y_1,'o')
plt.plot(x_2,y_2,'o')
plt.plot(x_3,y_3,'o')
plt.plot(x_4,y_4,'o')
plt.show() 

x = x_1+x_2+x_3+x_4
y = y_1+y_2+y_3+y_4
arr = pd.DataFrame([x,y]).T

from sklearn.cluster import KMeans

#change k value in n_clusters
kmean = KMeans(n_clusters = 4).fit(arr)
kmean.labels_
plt.scatter(arr.iloc[:,0], arr.iloc[:,1], c=kmean.labels_)

