import networkx as nx
import numpy as np
import os


def read_datasets():
    graphs = []
    Ks = []
    filenames = []
    for filename in os.listdir("data"):
        fn = "data/"+filename
        filenames.append(filename)
        if "K" in fn:
            Ks.append(fn.split("=")[1].split(".")[0])
        else:
            Ks.append(None)
        with open(fn, "r") as f:
            A = np.genfromtxt(f, delimiter=",", dtype=int)
            G = nx.from_numpy_matrix(A)
            graphs.append(G)
    return graphs, Ks, filenames
