import importlib
from sklearn.datasets import load_digits
import zlib
import json

import numpy as np

import run_ambreducer as ramb
importlib.reload(ramb)

from sklearn.decomposition import PCA

import matplotlib.pyplot as plt
import os
from tqdm import tqdm

from sklearn.preprocessing import StandardScaler


def read_dataset_by_path(path):
	path_data = path + "/data.bin"
	path_labels = path + "/label.bin"
	## open the data and label binary file
	with open(path_data, 'rb') as f:
		data_comp = f.read()
	with open(path_labels, 'rb') as f:
		labels_comp = f.read()
	## convert the data and label to np array
	data = np.array(json.loads(zlib.decompress(data_comp).decode('utf8')))
	labels = np.array(json.loads(zlib.decompress(labels_comp).decode('utf8')))

	return data, labels


# data, labels = read_dataset_by_path("../labeled-datasets/compressed/fashion_mnist/")
# raw = PCA(n_components=100).fit_transform(data)

method = "umap"
metric = "snc"
init_points = 10
n_iter = 50

dataset_list = os.listdir("../labeled-datasets/compressed/")
dataset_list = [x for x in dataset_list if x[-4:] != ".zip"]

used_datasets = []

for dataset in tqdm(dataset_list):

	if os.path.exists(f"./ambreducer_results/{dataset}_init.json"):
		continue
	data, labels = read_dataset_by_path("../labeled-datasets/compressed/" + dataset)

	if data.shape[0] > 5000:
		## downsample
		data = data[np.random.choice(data.shape[0], 5000, replace=False)]
	
	if data.shape[0] < 1500:
		continue
	data = StandardScaler().fit_transform(data)


	try:
		results = ramb.run(data, method=method, metric=metric, threshold_loss=0.05, S=3.0, verbose=2, init_points=init_points, n_iter=n_iter)
	except:
		continue

	init_emb = results["init_emb"]
	final_emb = results["final_emb"]	

	with open(f"./ambreducer_results/{dataset}_init.json", 'w') as f:
		json.dump(init_emb.tolist(), f)
	with open(f"./ambreducer_results/{dataset}_final.json", 'w') as f:
		json.dump(final_emb.tolist(), f)
	
	with open(f"./ambreducer_results/{dataset}_results.json", 'w') as f:
		del results["init_emb"]
		del results["final_emb"]
		json.dump(results, f)
	
	used_datasets.append(dataset)
