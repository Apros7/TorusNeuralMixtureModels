
import sys
sys.path.insert(0, '.')

from src.data.synthetic_data import sampleFromTorusGraph
from src.parameterEstimation.scoreMatching import SM
from src.parameterEstimation.trainNCE import mixture_torch_loop
from src.parameterEstimation.NCE import TorusGraphs

import numpy as np
import torch
import matplotlib.pyplot as plt
import concurrent.futures
import os
from functools import partialmethod
from tqdm import tqdm
import time

def sample(N, nodes, phi):
    X, datamodel = sampleFromTorusGraph(
        nodes = nodes,
        samples = N,
        phi = phi,
        fitFCM = False,
        fitPAD = True,
        fitPAS = False,
        return_datamodel = True
    )
    return X, datamodel

def NCE_estimate(X, N, nodes, K, lr = 0.1):
    X = torch.from_numpy(X).float().T
    noise = torch.rand(N,nodes)*2*torch.tensor(np.pi) 

    model = TorusGraphs(nodes=X.shape[1],K=K)
    model, objective = mixture_torch_loop(X,noise,model, lr= lr)

    theta = model.theta.detach().flatten().numpy()
    return lr, theta

def single_run(N, nodes, phi, K, lr_dict: dict):
    X, datamodel = sample(N, nodes, phi)

    with concurrent.futures.ProcessPoolExecutor(max_workers=None) as executor:
        futures = [executor.submit(NCE_estimate, X, N, nodes, K, lr = lr_i) for lr_i in lr_dict.keys()]
        results = [future.result() for future in concurrent.futures.as_completed(futures)]

    for lr_i, theta in results:
        dist_NCE = np.linalg.norm(phi - theta)
        lr_dict[lr_i].append(dist_NCE)
    return lr_dict

def cross_val_runs(cv_runs, N, nodes, phi, K, lr_dict):

    for _ in tqdm(range(cv_runs)):
        lr_dict = single_run(N, nodes, phi, K, lr_dict)

    return lr_dict

def plot(lr_dict):
    fig, ax = plt.subplots()
    boxplot_data = list(lr_dict.values())
    ax.boxplot(boxplot_data, patch_artist=True)
    ax.set_xticklabels(list(lr_dict.keys()))
    ax.set_title('Boxplot of NCE estimation with varying learning rate')
    plt.show()

def get_lr_dict(lrs_to_test):
    lr_dict = {}
    for v in lrs_to_test:
        lr_dict[v] = []
    return lr_dict

if __name__ == "__main__":
    os.environ['DISABLE_TQDM'] = 'True'
    start_time = time.time()

    N = 1000 # samples
    nodes = 3
    K = 1 # single model
    cv_runs = 4

    lrs_to_test = [10, 1, 0.5, 0.1, 0.05, 0.01, 0.005, 0.001]
    lr_dict = get_lr_dict(lrs_to_test)

    print(f"Estimated time: {40*cv_runs} sec")

    phi = np.block([ 0, 0, 8*np.cos(np.pi), 8*np.sin(np.pi), 0, 0 ])
    cross_val_runs(cv_runs, N, nodes, phi, K, lr_dict)
    print(f"Time taken = {time.time() - start_time}")
    plot(lr_dict)


