import torch
import numpy as np
from .riemannian_metric import RiemannianMetric
import tqdm
import networkx as nx
from sklearn.neighbors import NearestNeighbors


class RiemannianTree(object):
	def __init__(self, riemannian_metric):
		super(RiemannianTree, self).__init__()
		self.riemannian_metric = riemannian_metric

	def create(self, z, n_neighbors):
		n_data = len(z)
		knn = NearestNeighbors(n_neighbors = n_neighbors, metric = 'euclidean')
		knn.fit(z.detach().numpy())

		G = nx.Graph()

		#nodes
		for i in range(n_data):
			n_attr = {f'z{k}': float(z[i, k]) for k in range(z.shape[1])}
			G.add_node(i, **n_attr)

		#edges
		for i in tqdm.trange(n_data):
			distances, indices = knn.kneighbors(z.detach().numpy()[i:i+1])
			distances = distances[0]
			indices = indices[0]

			for ix, dist in zip(indices, distances):
				if (i, ix) in G.edges or (ix, i) in G.edges or i == ix:
					continue

				L_riemann,_ = self.riemannian_metric.riemannian_distance_along_line(i, ix)
				L_euclidean = dist

				edge_attr = {'weight': float(1/L_riemann),
						 	 'weight_euclidean': float(1/L_euclidean),
						 	 'distance_riemann': float(L_riemann),
						 	 'distance_euclidean': float(L_euclidean)}
				G.add_edge(i, ix, **edge_attr)

		return G
