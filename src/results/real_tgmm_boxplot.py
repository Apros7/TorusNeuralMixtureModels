import sys
sys.path.insert(0, '.')
from src.toolbox import load_sample_data, TorusGraph, NCE, sample_syndata_torusgraph
import torch
import matplotlib.pyplot as plt
import time
import numpy as np


if __name__ == "__main__":

    # os.environ['DISABLE_TQDM'] = 'True'
    start_time = time.time()

    N = 1000 # samples
    nodes = 3
    K = 3 # single model
    cv_runs = 10
    nce_steps = 1000
    lr = 0.05


    # data, true_vals = load_sample_data()
    # true_vals = torch.tensor(true_vals, dtype = torch.int64)
    # data = torch.from_numpy(data).float()

    X, datamodel = sample_syndata_torusgraph(
        nodes = nodes,
        samples = N,
        nModels = K,
        fitFCM = False,
        fitPAD = True,
        fitPAS = False,
        return_datamodel = True
    )

    accuracies = []
    for _ in range(cv_runs):
        estimation_method = NCE(
            nodes = nodes, 
            K = K, 
            lr = 0.1, 
            return_log_prop_data = True)

        torus_graph = TorusGraph(
            data = X,
            nodes = nodes, 
            samples = N,  
            nModels = K, 
            estimationMethod = estimation_method, 
            return_datamodel = True)
        accuracies.append(torus_graph.evaluate())
    print(f"Time taken = {time.time() - start_time}")
    print(np.mean(accuracies))
    print(np.std(accuracies))
    plt.rcParams['font.family'] = 'Times New Roman'
    plt.boxplot(accuracies, patch_artist=True)
    plt.title(f'Boxplot of acc on real data with learning rate = {lr} over {cv_runs} runs on Mixture model')
    plt.savefig('src/plots/tgmm_boxplot.png')
    plt.show()