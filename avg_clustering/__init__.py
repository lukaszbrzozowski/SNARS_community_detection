import numpy as np
from scipy.spatial import distance_matrix
from scipy.spatial.distance import pdist
from itertools import combinations
from tqdm import tqdm
import copy
def avg_clustering(X, k):
    clusts = {i: np.array([i]) for i in range(X.shape[0])}
    def avg_dist(i, j):
        clust_i = X[clusts[i[0]], :]
        clust_j = X[clusts[j[0]], :]
        dm = np.power(distance_matrix(clust_i, clust_j), 2)
        return np.average(dm)
    nclust = X.shape[0]
    while nclust > k:
        combs = np.array(list(clusts.keys())).reshape(-1, 1)
        combs1 = list(combinations(list(clusts.keys()),2))
        dm = pdist(combs, avg_dist)
        min_ix = np.where(dm == min(dm))[0][0]
        pair = combs1[min_ix]
        clusts[pair[0]] = np.append(clusts[pair[0]], clusts[pair[1]])
        del clusts[pair[1]]
        nclust = len(clusts.keys())
    return clusts

def get_values_for_gap_stat(X, clusts):
    k = len(clusts.keys())
    clust_map = {k : i for i, k in enumerate(clusts.keys())}
    new_clusts = {clust_map[k]: clusts[k] for k in clusts.keys()}

    cluster_centers = np.empty((k, X.shape[1]))
    arr_clusters = np.empty((X.shape[0]), dtype=np.int32)
    for j, i in enumerate(new_clusts.keys()):
        rows = new_clusts[i]
        cur_cluster = X[rows, :]
        centroid = np.average(cur_cluster, axis=0)
        cluster_centers[j, :] = centroid
        arr_clusters[rows] = i

    return cluster_centers, arr_clusters




