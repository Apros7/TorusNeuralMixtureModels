from src.data.synthetic_data import sampleFromTorusGraph
from src.parameterEstimation.NCE import NCE
from src.parameterEstimation.scoreMatching import SM

class TorusGraph():
    def __init__(self, data, estimation_func: NCE or SM = NCE): # estimation_func can be NCE or SM
        self.data = data
        self.phi, self.buffer = self.estimate_params(estimation_func)
        self.nodes = 0
        self.nodePairs = 0
        
    def estimate_phi(self):
        self.phi, self.c = "hey", "hey"

    def sample(self, samples : int = 100):
        return sampleFromTorusGraph(self.nodes, samples, phi = self.phi, nodePairs=self.nodePairs)
    



import sys
sys.path.insert(0, '.')

from src.toolbox import TorusGraph, NCE, sample_syndata_torusgraph, TorusGraphInformation
import numpy as np
import torch
import matplotlib.pyplot as plt
import concurrent.futures
from tqdm import tqdm
import time
from typing import Tuple

def sample(N, nodes, phi, nModels) -> Tuple[np.ndarray, TorusGraphInformation]:
    X, datamodel = sample_syndata_torusgraph(
        nodes = nodes,
        samples = N,
        phi = phi,
        nModels = nModels,
        fitFCM = False,
        fitPAD = True,
        fitPAS = False,
        return_datamodel = True
    )
    return X, datamodel

def NCE_estimate(X, N, nodes, K, step = 0.1):
    X = torch.from_numpy(X).float().T
    estimation_method = NCE(nodes = X.shape[1], K = K, lr = 0.1, steps = step, return_log_prop_data = True)
    torus_graph = TorusGraph(nodes = X.shape[1], samples = X.shape[0], data = X, nModels = K, estimationMethod = estimation_method)

    theta = estimation_method.theta.detach().flatten().numpy()
    return [step, theta, torus_graph]

def single_run(N, nodes, phi, K, steps_dict: dict):
    X, datamodel = sample(N, nodes, phi, nModels = K)
    phi = list(datamodel.phi)

    with concurrent.futures.ProcessPoolExecutor(max_workers=None) as executor:
        futures = [executor.submit(NCE_estimate, X, N, nodes, K, step = step) for step in steps_dict.keys()]
        results = [future.result() for future in concurrent.futures.as_completed(futures)]

    for step, theta, torus_graph in results:
        if K == 1:
            value = np.linalg.norm(phi - theta.reshape((K, -1)))
        else:
            value = torus_graph.evaluate()
        steps_dict[step].append(value)
        
    return steps_dict

def cross_val_runs(cv_runs, N, nodes, phi, K, steps_dict):

    for _ in tqdm(range(cv_runs)):
        lr_dict = single_run(N, nodes, phi, K, steps_dict)

    return lr_dict

def plot(lr_dict, title):
    plt.rcParams['font.family'] = 'Times New Roman' 
    fig, ax = plt.subplots()
    boxplot_data = list(lr_dict.values())
    ax.boxplot(boxplot_data, patch_artist=True)
    ax.set_xticklabels(list(lr_dict.keys()))
    ax.set_title(title)

if __name__ == "__main__":
    # os.environ['DISABLE_TQDM'] = 'True'
    start_time = time.time()

    N = 1000 # samples
    nodes = 3
    K = 3 # single model
    cv_runs = 10

    steps_to_test = [50, 200, 500, 1000, 2000, 5000]
    steps_dict = {lr: [] for lr in steps_to_test}

    print(f"Estimated time: {40*cv_runs} sec")

    phi = None # you can specify here if you want
    steps_dict = cross_val_runs(cv_runs, N, nodes, phi, K, steps_dict)
    print(f"Time taken = {time.time() - start_time}")
    print(steps_dict)
    plot(steps_dict, 'Boxplot of NCE estimation with varying steps')
    plt.xlabel('Steps')
    plt.ylabel('NMI')
    plt.savefig('src/plots/NCE_syn_steps.png')
    plt.show()


import numpy as np
import pyTG
import matplotlib.pyplot as plt

X = np.random.rand(3, 100)*2*np.pi

out = pyTG.torusGraphs(X, selMode=(False, True, False)) #en samling af variable, som er NodeGraphs, kig på 5 som er phihat
phi_hat = out[-2]

# Given arrays
Phi_cos = np.zeros((3,3))
Phi_sin = np.zeros((3,3))
print(phi_hat)

# Given values to assign
phi_hat = [phi_hat[0], phi_hat[1], phi_hat[2], phi_hat[3], phi_hat[4], phi_hat[5]] 

# List of assignments: (i, j, phi_hat_index, matrix)
assignments = [
    (0, 1, 0, Phi_cos),
    (0, 2, 1, Phi_sin),
    (1, 2, 2, Phi_cos),
    (0, 1, 3, Phi_sin),
    (0, 2, 4, Phi_cos),
    (1, 2, 5, Phi_sin)
]

# For loop to perform the assignments
for i, j, index, matrix in assignments:
    matrix[i, j] = phi_hat[index]
print(matrix)

plt.figure()
fig,axs = plt.subplots(1,2)
pos = axs[0].imshow(Phi_cos)
fig.colorbar(pos, ax=axs[0])
pos = axs[1].imshow(Phi_sin)
fig.colorbar(pos, ax=axs[1])
plt.savefig('plot_sm.png')

a=7

#matricer symetrisk - kun værdier i den øvere del eller nedre del transporneret
#farve hvor stærk fasekoblingen er, viser de forskellige dele af hjerne området