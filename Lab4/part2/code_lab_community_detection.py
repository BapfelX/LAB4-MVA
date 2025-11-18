"""
Graph Mining - ALTEGRAD - Nov 2024
"""

import networkx as nx
import numpy as np
from scipy.sparse.linalg import eigs
from scipy.sparse import diags, eye
from random import randint
from sklearn.cluster import KMeans



############## Task 3
def spectral_clustering(G, k):
    M = nx.adjacency_matrix(G).todense()  # conversion en matrice dense
    n = G.number_of_nodes()
    D = np.zeros((n,n))
    for i in range(n):
        D[i,i] = np.sum(M[i,:])
    
    L = np.identity(n) - np.linalg.inv(D) @ M
    eigvals, eigvecs = np.linalg.eig(L)
    
    idx = np.argsort(eigvals)[:k]
    U = np.real(eigvecs[:, idx])  
    kmeans = KMeans(n_clusters=k, random_state=42)
    clustering = kmeans.fit_predict(U)
    
    return clustering

############## Task 4

G=nx.read_edgelist("/users/eleves-b/2022/felix.rosseeuw/LAB4-MVA/Lab4/datasets/CA-HepTh.txt")

clustering=spectral_clustering(G,k=50)

print(clustering[0:10])




############## Task 5
# Compute modularity value from graph G based on clustering
def modularity(G, clustering):
    nodes = list(G.nodes())
    index = {node: i for i, node in enumerate(nodes)}
    k=np.max(clustering)+1
    m=G.number_of_edges()
    L=np.zeros(k)
    U=np.zeros(k)
    for (u,v) in G.edges() :
        if clustering[index[u]]==clustering[index[v]] :
            L[clustering[index[u]]]+=1
    for i, node in enumerate(nodes):
        U[clustering[i]] += G.degree(node)
  
    return(np.sum(L/m-U*U/(4*m**2)))


    
    
    



############## Task 6

print(modularity(G, clustering))

clustering=np.array([randint(0,50) for i in range (G.number_of_nodes())])

print(modularity(G, clustering))





