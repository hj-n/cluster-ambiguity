
import hdbscan
from sklearn.cluster import DBSCAN, KMeans, Birch, AgglomerativeClustering
from sklearn_extra.cluster import KMedoids
from pyclustering.cluster.xmeans import xmeans 

from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score

from bayes_opt import BayesianOptimization
import numpy as np

def silhouette_scorer(X, clustering):
  try:
	  return silhouette_score(X, clustering)
  except:
	  return -1

def calinski_harabasz_scorer(X, clustering):
	try:
		return calinski_harabasz_score(X, clustering)
	except:
		return -1

def davies_bouldin_scorer(X, clustering):
	try:
		return davies_bouldin_score(X, clustering)
	except:
		return -1
	
def hdbscan_naive_scorer(X, scorer):
	hdbscan_clusterer = hdbscan.HDBSCAN()
	hdbscan_clusterer.fit(X)
	return scorer(X, hdbscan_clusterer.labels_)

def dbscan_naive_scorer(X, scorer):
	dbscan_clusterer = DBSCAN()
	dbscan_clusterer.fit(X)
	return scorer(X, dbscan_clusterer.labels_)

def kmeans_naive_scorer(X, scorer):
	kmeans_clusterer = KMeans()
	kmeans_clusterer.fit(X)
	return scorer(X, kmeans_clusterer.labels_)

def birch_naive_scorer(X, scorer):
	birch_clusterer = Birch()
	birch_clusterer.fit(X)
	return scorer(X, birch_clusterer.labels_)

def agglo_complete_naive_scorer(X, scorer):
	return agglomerative_naive_scorer(X, scorer, linkage="complete")

def agglo_average_naive_scorer(X, scorer):
	return agglomerative_naive_scorer(X, scorer, linkage="average")

def agglo_single_naive_scorer(X, scorer):
	return agglomerative_naive_scorer(X, scorer, linkage="single")

def agglomerative_naive_scorer(X, scorer, linkage):
	agglomerative_clusterer = AgglomerativeClustering(linkage=linkage)
	agglomerative_clusterer.fit(X)
	return scorer(X, agglomerative_clusterer.labels_)

def kmedoids_naive_scorer(X, scorer):
	kmedoids_clusterer = KMedoids()
	kmedoids_clusterer.fit(X)
	return scorer(X, kmedoids_clusterer.labels_)

def xmeans_naive_scorer(X, scorer):
	xmeans_clusterer = xmeans(X)
	xmeans_clusterer.process()
	return scorer(X, xmeans_clusterer.get_clusters())


def hdbscan_scorer(X, scorer):
	def hdbscan_scorer_inner(min_cluster_size, min_samples, cluster_selection_epsilon):
		min_cluster_size = int(min_cluster_size)
		min_samples = int(min_samples)
		cluster_selection_epsilon = float(cluster_selection_epsilon)
		hdbscan_clusterer = hdbscan.HDBSCAN(min_cluster_size=min_cluster_size, min_samples=min_samples, cluster_selection_epsilon=cluster_selection_epsilon)
		hdbscan_clusterer.fit(X)
		return scorer(X, hdbscan_clusterer.labels_)

	pbounds = {
		'min_cluster_size': (2, 50),
		'min_samples': (1, 10),
		'cluster_selection_epsilon': (0.01, 1.0)
	}

	optimizer = BayesianOptimization(hdbscan_scorer_inner, pbounds, verbose=0, random_state=1)
	optimizer.maximize()
	
	return optimizer.max['target']


def dbscan_scorer(X, scorer):
	def dbscan_scorer_inner(eps, min_samples):
		eps = float(eps)
		min_samples = int(min_samples)
		dbscan_clusterer = DBSCAN(eps=eps, min_samples=min_samples)
		dbscan_clusterer.fit(X)
		return scorer(X, dbscan_clusterer.labels_)
	
	pbounds = {
		'eps': (0.01, 1.0),
		'min_samples': (1, 10)
	}

	optimizer = BayesianOptimization(f=dbscan_scorer_inner, pbounds=pbounds, verbose=0, random_state=1)
	optimizer.maximize()

	return optimizer.max['target']

def kmeans_scorer(X, scorer):
	def kmeans_scorer_inner(n_clusters):
		n_clusters = int(n_clusters)
		kmeans_clusterer = KMeans(n_clusters=n_clusters)
		kmeans_clusterer.fit(X)
		return scorer(X, kmeans_clusterer.labels_)
	
	pbounds = {
		"n_clusters": (2, 3 * int(np.sqrt(X.shape[0])))
	}

	optimizer = BayesianOptimization(f=kmeans_scorer_inner, pbounds=pbounds, verbose=0, random_state=1)
	optimizer.maximize()

	return optimizer.max['target']

def kmedoid_scorer(X, scorer):
	def kmedoid_scorer_inner(n_clusters):
		n_clusters = int(n_clusters)
		kmedoid_clusterer = KMedoids(n_clusters=n_clusters)
		kmedoid_clusterer.fit(X)
		return scorer(X, kmedoid_clusterer.labels_)
	
	pbounds = {
		"n_clusters": (2, 3 * int(np.sqrt(X.shape[0])))
	}

	optimizer = BayesianOptimization(f=kmedoid_scorer_inner, pbounds=pbounds, verbose=0, random_state=1)
	optimizer.maximize()

	return optimizer.max['target']

def xmeans_scorer(X, scorer):
	def xmeans_scorer_inner(kmax, tolerance):
		kmax = int(kmax)
		tolerance = float(tolerance)
		xmeans_clusterer = xmeans(X, kmax=kmax, tolerance=tolerance)
		xmeans_clusterer.process()
		return scorer(X, xmeans_clusterer.get_clusters())
	
	pbounds = {
		"kmax": (2, 50),
		"tolerance": (0.01, 1.0)
	}

	optimizer = BayesianOptimization(f=xmeans_scorer_inner, pbounds=pbounds, verbose=0, random_state=1)
	optimizer.maximize()

	return optimizer.max['target']

def birch_scorer(X, scorer):
	def birch_scorer_inner(threshold, branching_factor):
		threshold = float(threshold)
		branching_factor = int(branching_factor)
		birch_clusterer = Birch(threshold=threshold, branching_factor=branching_factor)
		birch_clusterer.fit(X)
		return scorer(X, birch_clusterer.labels_)
	
	pbounds = {
		"threshold": (0.01, 1.0),
		"branching_factor": (10, 100)
	}
	optimizer = BayesianOptimization(f=birch_scorer_inner, pbounds=pbounds, verbose=0, random_state=1)
	optimizer.maximize()

	return optimizer.max['target']

def agglo_complete_scorer(X, scorer):
	return agglo_scorer(X, scorer, linkage='complete')

def agglo_average_scorer(X, scorer):
	return agglo_scorer(X, scorer, linkage='average')

def agglo_single_scorer(X, scorer):
	return agglo_scorer(X, scorer, linkage='single')

def agglo_scorer(X, scorer, linkage):
	def agglo_scorer_inner(n_clusters):
		n_clusters = int(n_clusters)
		agglo_clusterer = AgglomerativeClustering(n_clusters=n_clusters, linkage=linkage)
		agglo_clusterer.fit(X)
		return scorer(X, agglo_clusterer.labels_)
	
	pbounds = {
		"n_clusters": (2, 3 * int(np.sqrt(X.shape[0])))
	}

	optimizer = BayesianOptimization(f=agglo_scorer_inner, pbounds=pbounds, verbose=0, random_state=1)
	optimizer.maximize()

	return optimizer.max['target']

