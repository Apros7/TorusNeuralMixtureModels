
import sys
sys.path.insert(0, '.')

from src.results.NCE_lr import plot, cross_val_runs
import time
import os


if __name__ == "__main__":
    # os.environ['DISABLE_TQDM'] = 'True'
    start_time = time.time()

    N = 1000 # samples
    nodes = 3
    K = 3 # single model
    cv_runs = 4
    nce_steps = 2000

    lrs_to_test = [10, 1, 0.5, 0.1, 0.05, 0.01, 0.005, 0.001]
    lr_dict = {lr: [] for lr in lrs_to_test}

    print(f"Estimated time: {40*cv_runs} sec")

    phi = None # you can specify here if you want
    lr_dict = cross_val_runs(cv_runs, N, nodes, phi, K, lr_dict, nce_steps)
    print(f"Time taken = {time.time() - start_time}")
    # print(lr_dict)
    plot(lr_dict, 'Boxplot of NCE estimation with varying learning rate on Mixture model')


