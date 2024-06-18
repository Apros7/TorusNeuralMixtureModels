
import sys
sys.path.insert(0, '.')

from src.toolbox import NCE, TorusGraph, sample_syndata_torusgraph
import time
import os
import matplotlib.pyplot as plt
import numpy as np


if __name__ == "__main__":
    # os.environ['DISABLE_TQDM'] = 'True'
    start_time = time.time()

    N = 1000 # samples
    nodes = 3
    cv_runs = 5
    nce_steps = 2000

    Ks = [1, 2, 3, 4, 5]
    losses_dict = {K: [] for K in Ks}
    
    for _ in range(cv_runs): 

        data, datainformation = sample_syndata_torusgraph(
            nodes = nodes,
            samples = N,
            nModels = 3,
            return_datamodel = True,
        )

        for K in Ks:
            nce = NCE(
                nodes = nodes,
                K = K,
                steps = nce_steps,
                lr = 0.1,
            )
            TorusGraph(
                data = data,
                TGInformation = datainformation,
                nodes = nodes,
                samples = N,
                nModels = K,
                estimationMethod = nce,
            )
            losses_dict[K].append(nce.losses[-1])
    boxplot_data = list(losses_dict.values())
    fig, ax = plt.subplots()
    ax.boxplot(boxplot_data, patch_artist=True)
    ax.set_xticklabels(list(losses_dict.keys()))
    # ax.title(f"NCE loss over nModels sampled on 3 torus graphs")
    plt.show()