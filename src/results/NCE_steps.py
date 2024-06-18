
import sys
sys.path.insert(0, '.')

from src.toolbox import TorusGraph, NCE, SM, sample_syndata_torusgraph, TorusGraphInformation

import numpy as np
import torch
import matplotlib.pyplot as plt
import concurrent.futures
import os
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
    fig, ax = plt.subplots()
    boxplot_data = list(lr_dict.values())
    ax.boxplot(boxplot_data, patch_artist=True)
    ax.set_xticklabels(list(lr_dict.keys()))
    ax.set_title(title)
    plt.show()

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


