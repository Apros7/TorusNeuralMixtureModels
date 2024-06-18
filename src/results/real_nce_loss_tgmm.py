import sys
sys.path.insert(0, '.')
from src.toolbox import load_sample_data, TorusGraph, NCE
import time
import torch
import matplotlib.pyplot as plt
import time


if __name__ == "__main__":

    # os.environ['DISABLE_TQDM'] = 'True'
    start_time = time.time()

    N = 1000 # samples
    nodes = 3
    K = 3 # single model
    cv_runs = 2
    nce_steps = 2000
    lr = 0.1

    data, true_vals = load_sample_data()
    data = torch.from_numpy(data).float()
    estimation_method = NCE(nodes = data.shape[1], K = K, lr = lr, steps = nce_steps, return_log_prop_data = True)
    torus_graph = TorusGraph(nodes = data.shape[1], samples = data.shape[0], data = data, nModels = K, estimationMethod = estimation_method, true_vals = true_vals)
    print(f"Time taken = {time.time() - start_time}")
    plt.rcParams['font.family'] = 'Times New Roman'
    plt.plot(estimation_method.losses)
    plt.title(f'NCE loss for learning rate = {lr} on Mixture model with real data')
    plt.savefig('src/plots/real_nce_loss_tgmm.png')
    plt.show()
