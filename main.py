import relegy.embeddings as rle
import networkx as nx
import numpy as np
from avg_clustering import avg_clustering, get_values_for_gap_stat
from read_data import read_datasets
from gap_statistic.optimalK import OptimalK
import time
from tqdm import tqdm

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
    graphs, Ks, filenames = read_datasets()
    exec_times = []
    results_path = "Lukasz_Brzozowski/"
    for i, G in tqdm(enumerate(graphs)):
        filename = filenames[i]
        k = Ks[i]
        if k is not None:
            k = int(k)
            start = time.time()
            clustering = cluster(G, k)
            finish = time.time()
            exec_time = finish-start
            exec_times.append(exec_time)
        else:
            start = time.time()
            clustering = cluster(G)
            finish = time.time()
            exec_time=finish-start
            exec_times.append(exec_time)
        with open(results_path+filename, "w") as f:
            for i, r in enumerate(clustering):
                f.write(f"{i+1}, {clustering[i]} \n")
    with open("Lukasz_Brzozowski/description.txt", "w") as f:
        f.write("Lukasz Brzozowski\n")
        f.write("https://github.com/lukaszbrzozowski/SNARS_community_detection\n")
        for i in range(len(exec_times)):
            f.write(filenames[i] + ", " + str(exec_times[i]) +"\n")