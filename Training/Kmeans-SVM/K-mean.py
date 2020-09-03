from __future__ import print_function
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.distance import cdist
import random
np.random.seed(18)
from p5 import *


means = [[1,1], [9,8], [-2,-4]]
cov = [[1,0], [0,1]]

N = 500
X0 = np.random.multivariate_normal(means[0], cov, N)
X1 = np.random.multivariate_normal(means[1], cov, N)
X2 = np.random.multivariate_normal(means[2], cov, N)
X = np.concatenate((X0, X1, X2), axis=0)
K = int(X.shape[0]/N)
original_label = np.array([0]*N + [1]*N + [2]*N).T




def kmeans_init_centroids(X, k):
    return X[np.random.choice(X.shape[0], k, replace=False)]

def kmeans_assign_lables(X, centroids):
    D = cdist(X, centroids)
    return np.argmin(D, axis=1)

def has_converged(centroids, new_centroids):
    return (set([tuple(a) for a in centroids]) == 
    set([tuple(a) for a in new_centroids]))

def kmeans_update_centroids(X, lables, K):
    centroids = np.zeros((K, X.shape[1]))
    for k in range(K):
        Xk = X[lables == k,:]
        centroids[k,:] = np.mean(Xk, axis = 0)
    return centroids

def kmeans(X, K):
    centroids = [kmeans_init_centroids(X, K)]
    labels = []
    it = 0
    while True:
        labels.append(kmeans_assign_lables(X, centroids[-1]))
        new_centroids = kmeans_update_centroids(X, labels[-1], K)
        if has_converged(centroids[-1], new_centroids):
            break
        centroids.append(new_centroids)
        it += 1
    return (centroids, labels, it)

centroids, labels, it = kmeans(X, K)


plt.scatter(X[labels[-1] == 0,0], X[labels[-1] == 0,1], s =5)
plt.scatter(X[labels[-1] == 1,0], X[labels[-1] == 1,1], s = 5)
plt.scatter(X[labels[-1] == 2,0], X[labels[-1] == 2,1], s = 5)
plt.scatter(centroids[-1][:,0], centroids[-1][:,1], s=50, marker='*')


plt.show()