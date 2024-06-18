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


if __name__ == "__main__":

    # os.environ['DISABLE_TQDM'] = 'True'
    start_time = time.time()

    N = 1000 # samples
    nodes = 3
    K = 3 # single model
    cv_runs = 10

    plt.rcParams['font.family'] = 'Times New Roman'
    fig, axs = plt.subplots(2, 2)
    X, datamodel = sample_syndata_torusgraph(
        nodes = nodes,
        samples = N,
        nModels = K,
        fitFCM = False,
        fitPAD = True,
        fitPAS = False,
        return_datamodel = True
    )

    axs = [x for ax in axs for x in ax]

    for i, steps in enumerate([500, 2000, 5000, 10000]):

        estimation_method = NCE(
            nodes = nodes, 
            K = K, 
            lr = 0.1, 
            steps = steps, 
            return_log_prop_data = True)

        torus_graph = TorusGraph(
            data = X,
            nodes = nodes, 
            samples = N,  
            nModels = K, 
            estimationMethod = estimation_method, 
            return_datamodel = True)

        torus_graph.visualize(
            title = f"Dist of preds after {steps} NCE steps",
            ax = axs[i],
            show = False
        )
        
    plt.tight_layout()
    plt.savefig('src/plots/syn_preds_dist.png')
    plt.show()