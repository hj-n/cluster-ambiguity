import numpy as np
import pandas as pd
import CLAMS as clams
from tqdm import tqdm
from bayes_opt import BayesianOptimization
from scipy.stats import spearmanr
import random


keys = np.load("../scatterplots/userstudy_ambiguity/keys.npy")
ami_amb = np.load("../scatterplots/userstudy_ambiguity/ami_ambiguity_list.npy")
arand_amb = np.load("../scatterplots/userstudy_ambiguity/arand_ambiguity_list.npy")
vm_amb = np.load("../scatterplots/userstudy_ambiguity/vm_ambiguity_list.npy")
homo_amb = np.load("../scatterplots/userstudy_ambiguity/homo_ambiguity_list.npy")
comp_amb = np.load("../scatterplots/userstudy_ambiguity/comp_ambiguity_list.npy")

# keys = keys[:3]
# ami_amb = ami_amb[:3]
# arand_amb = arand_amb[:3]
# vm_amb = vm_amb[:3]
# homo_amb = homo_amb[:3]
# comp_amb = comp_amb[:3]



for i in range(100):
	S = random.uniform(0, 1) * 9 + 1
	cs = clams.ClusterAmbiguity(mode="entropy", S=S)
	amb_list = []
	for j, dataset in tqdm(enumerate(keys)):
		splot = np.load(f"../scatterplots/scatterplots/{dataset}")
		try:
			ambiguity = cs.fit(splot)
			amb_list.append(ambiguity)
		except:
			amb_list.append(np.nan)
	amb_list = np.array(amb_list)

	np.save(f"../scatterplots/sampling/ambiguity_{i}_{S}.npy", amb_list)

	nan_arr = np.isnan(amb_list)

	amb_list = amb_list[~nan_arr]
	ami_amb_wo_nan = ami_amb[~nan_arr]
	arand_amb_wo_nan = arand_amb[~nan_arr]
	vm_amb_wo_nan = vm_amb[~nan_arr]
	homo_amb_wo_nan = homo_amb[~nan_arr]
	comp_amb_wo_nan = comp_amb[~nan_arr]

	ami_spearman = spearmanr(amb_list, ami_amb_wo_nan)
	arand_spearman = spearmanr(amb_list, arand_amb_wo_nan)
	vm_spearman = spearmanr(amb_list, vm_amb_wo_nan)
	homo_spearman = spearmanr(amb_list, homo_amb_wo_nan)
	comp_spearman = spearmanr(amb_list, comp_amb_wo_nan)

	print(f"S: {S}")
	print(f"AMI: {ami_spearman}", end=" ")
	print(f"ARAND: {arand_spearman}", end=" ")
	print(f"VM: {vm_spearman}", end=" ")
	print(f"HOMO: {homo_spearman}", end=" ")
	print(f"COMP: {comp_spearman}")
	print("=====================================")



# keys = keys[::5]
# ami_amb = ami_amb[::5]
# arand_amb = arand_amb[::5]
# vm_amb = vm_amb[::5]
# homo_amb = homo_amb[::5]
# comp_amb = comp_amb[::5]
