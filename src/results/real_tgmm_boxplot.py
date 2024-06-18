import sys
sys.path.insert(0, '.')

from src.toolbox import load_sample_data, TorusGraph, NCE

import os
import time
import numpy as np
import torch
import matplotlib.pyplot as plt
import concurrent.futures
import os
from tqdm import tqdm
import time


if __name__ == "__main__":

    # os.environ['DISABLE_TQDM'] = 'True'
    start_time = time.time()

    N = 1000 # samples
    nodes = 3
    K = 3 # single model
    cv_runs = 2
    nce_steps = 5000
    lr = 0.1


    data, true_vals = load_sample_data()
    data = torch.from_numpy(data).float()
    accuracies = []
    for _ in range(cv_runs):
        estimation_method = NCE(nodes = data.shape[1], K = K, lr = lr, steps = nce_steps, return_log_prop_data = True)
        torus_graph = TorusGraph(nodes = data.shape[1], samples = data.shape[0], data = data, nModels = K, estimationMethod = estimation_method, true_vals = true_vals)
        accuracies.append(torus_graph.evaluate())
    print(f"Time taken = {time.time() - start_time}")
    plt.boxplot(accuracies, patch_artist=True)
    plt.title(f'Boxplot of acc on real data with learning rate = {lr} over {cv_runs} runs on Mixture model')
    plt.show()