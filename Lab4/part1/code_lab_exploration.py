"""
Graph Mining - ALTEGRAD - Nov 2024
"""

import networkx as nx
import matplotlib.pyplot as plt
import numpy as np


############## Task 1

G=nx.read_edgelist("/users/eleves-b/2022/felix.rosseeuw/LAB4-MVA/Lab4/datasets/CA-HepTh.txt")

print(f"Total number of nodes: {G.number_of_nodes()}")
print(f"Total number of edges: {G.number_of_edges()}")
############## Task 2

S=nx.connected_components(G)

largest_cc = max(S, key=len)

print(f"number of nodes of the largest subset: {len(largest_cc)}")
print(f"percentage of nodes of the largest subset: {len(largest_cc)/G.number_of_nodes()*100}")

G_largest = G.subgraph(largest_cc)

print(f"Number of edges in the largest component: {G_largest.number_of_edges()}")
print(f"percentage of edges of the largest subset: {G_largest.number_of_edges()/G.number_of_edges()*100}")


