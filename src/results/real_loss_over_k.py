
import sys
sys.path.insert(0, '.')

from src.toolbox import NCE, TorusGraph, load_sample_data
import time
import os
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
import torch


if __name__ == "__main__":
    # os.environ['DISABLE_TQDM'] = 'True'
    start_time = time.time()

    N = 1000 # samples
    nodes = 3
    cv_runs = 5
    nce_steps = 2000

    Ks = [1, 2, 3, 4, 5]
    losses_dict = {K: [] for K in Ks}


    data, true_vals = load_sample_data()
    data = torch.from_numpy(data).float()

    for _ in tqdm(range(cv_runs)): 
        for K in Ks:
            nce = NCE(
                nodes = data.shape[1],
                K = K,
                steps = nce_steps,
                lr = 0.1,
            )
            TorusGraph(
                data = data,
                nodes = data.shape[1], 
                samples = data.shape[0], 
                nModels = K,
                estimationMethod = nce,
                true_vals = true_vals
            )
            losses_dict[K].append(nce.losses[-1])
    boxplot_data = list(losses_dict.values())
    fig, ax = plt.subplots()
    ax.boxplot(boxplot_data, patch_artist=True)
    ax.set_xticklabels(list(losses_dict.keys()))
    # ax.title(f"NCE loss over nModels sampled on 3 torus graphs")
    plt.show()