import sys
sys.path.insert(0, '.')
from src.toolbox import load_sample_data
from src.results.NCE_lr import cross_val_runs, plot
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
    cv_runs = 10
    nce_steps = 5000

    lrs_to_test = [10, 1, 0.5, 0.1, 0.05, 0.01, 0.005, 0.001]
    lr_dict = {lr: [] for lr in lrs_to_test}

    data, true_vals = load_sample_data()
    data = torch.from_numpy(data).float()
    phi = None
    lr_dict = cross_val_runs(cv_runs, N, nodes, phi, K, lr_dict, nce_steps)
    print(f"Time taken = {time.time() - start_time}")
    plot(lr_dict, 'Boxplot of NCE estimation with varying learning rate on Mixture model')
    plt.savefig('src/plots/real_NCE_lr_tgmm.png')
    plt.show()