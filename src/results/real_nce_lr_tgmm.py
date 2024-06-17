import sys
sys.path.insert(0, '.')

from src.toolbox import load_sample_data, TorusGraph, NCE, sample_syndata_torusgraph

import os
import time
import numpy as np
import torch
import matplotlib.pyplot as plt
import concurrent.futures
import os
from tqdm import tqdm
import time

# def NCE_estimate(X, N, nodes, K, lr = 0.1, nce_steps = 2000, true_vals = None):
#     X = torch.from_numpy(X).float().T


#     theta = estimation_method.theta.detach().flatten().numpy()
#     return [lr, theta, torus_graph]

# def single_run(N, nodes, phi, K, lr_dict: dict, nce_steps: int, X, true_vals):

#     with concurrent.futures.ProcessPoolExecutor(max_workers=None) as executor:
#         futures = [executor.submit(NCE_estimate, X, N, nodes, K, lr = lr_i, nce_steps = nce_steps, true_vals = true_vals) for lr_i in lr_dict.keys()]
#         results = [future.result() for future in concurrent.futures.as_completed(futures)]

#     for lr_i, theta, torus_graph in results:
#         if K == 1:
#             value = np.linalg.norm(phi - theta.reshape((K, -1)))
#         else:
#             value = torus_graph.evaluate()
#         lr_dict[lr_i].append(value)
        
#     return lr_dict

# def cross_val_runs(cv_runs, N, nodes, phi, K, lr_dict, nce_steps):

#     for _ in tqdm(range(cv_runs)):
#         lr_dict = single_run(N, nodes, phi, K, lr_dict, nce_steps)

#     return lr_dict

# def plot(lr_dict, title):
#     fig, ax = plt.subplots()
#     boxplot_data = list(lr_dict.values())
#     ax.boxplot(boxplot_data, patch_artist=True)
#     ax.set_xticklabels(list(lr_dict.keys()))
#     ax.set_title(title)
#     plt.show()


if __name__ == "__main__":

    # os.environ['DISABLE_TQDM'] = 'True'
    start_time = time.time()

    N = 1000 # samples
    nodes = 3
    K = 3 # single model
    cv_runs = 10
    nce_steps = 5000

    lrs_to_test = [10, 1, 0.5, 0.1, 0.05, 0.01, 0.005, 0.001]
    lr_dict = {lr: [] for lr in lrs_to_test}

    data, true_vals = load_sample_data()
    data = torch.from_numpy(data).float()
    # data, tginf = sample_syndata_torusgraph(nodes = nodes, samples = N, nModels = K, return_datamodel = True)
    # data = torch.from_numpy(data).float().T
    # print(data.shape)
    # true_vals = None

    estimation_method = NCE(
        nodes = data.shape[1], 
        K = K, 
        lr = 0.1, 
        steps = nce_steps, 
        return_log_prop_data = True)

    torus_graph = TorusGraph(
        nodes = data.shape[1], 
        samples = data.shape[0], 
        data = data, 
        nModels = K, 
        estimationMethod = estimation_method, 
        true_vals = true_vals)

    value = torus_graph.evaluate()