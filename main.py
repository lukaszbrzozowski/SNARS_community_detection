import relegy.embeddings as rle
import networkx as nx
import numpy as np
from avg_clustering import avg_clustering, get_values_for_gap_stat
from gap_statistic.optimalK import OptimalK

def util_clustering_function(X, k):
    clustering = avg_clustering(X, k)
    return get_values_for_gap_stat(X, clustering)

def find_optimal_k(X):
    optimalk = OptimalK(clusterer=util_clustering_function, n_jobs=8, parallel_backend='joblib')
    n = X.shape[0]
    n_clusters = optimalk(X, cluster_array=range(1, 5))
    return n_clusters

def cluster(G: nx.Graph, k = None):
    n = G.number_of_nodes()
    d = int(min(n, 2*np.log2(n)))
    embed = rle.DNGR.fast_embed(G,
                            d=d,
                            n_layers=2,
                            n_hid=[n, d],
                            dropout=0.05,
                            num_iter=50)
    if k is not None:
        print("Clustering...")
        clustering = avg_clustering(embed, k)
        return get_values_for_gap_stat(embed, clustering)[1]+1
    else:
        print("Searching for best k...")
        k = find_optimal_k(embed)
        print(f"Found k={k}")
        print("Clustering...")
        clustering = avg_clustering(embed, k)
        return get_values_for_gap_stat(embed, clustering)[1] + 1


if __name__ == '__main__':
    G = nx.stochastic_block_model([50, 50, 50], [[0.9, 0.1, 0.1], [0.1, 0.9, 0.1], [0.1, 0.1, 0.9]])
    cm = cluster(G)
    print(cm)