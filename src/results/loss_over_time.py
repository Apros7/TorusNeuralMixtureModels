
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
    cv_runs = 2
    nce_steps = 75

    data, datainformation = sample_syndata_torusgraph(
        nodes = nodes,
        samples = N,
        nModels = 3,
        return_datamodel = True,
    )
    losses_dict = {}

    for K in [1, 2, 3, 4, 5]:
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
        losses_dict[K] = nce.losses
    print(losses_dict)
    for k, v in losses_dict.items():
        plt.plot(v, label=f'k = {k}')
    # plt.plot(list(losses_dict.keys()), list(losses_dict.values()))
    plt.title(f"NCE loss over nModels sampled on 3 torus graphs")
    plt.legend()
    plt.show()