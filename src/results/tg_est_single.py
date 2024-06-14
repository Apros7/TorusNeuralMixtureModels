
import sys
sys.path.insert(0, '.')

from src.data.synthetic_data import sampleFromTorusGraph
from src.parameterEstimation.scoreMatching import SM
from src.parameterEstimation.trainNCE import mixture_torch_loop
from src.parameterEstimation.NCE import TorusGraphs

import numpy as np
import torch
import matplotlib.pyplot as plt
import concurrent.futures
import os
from functools import partialmethod
import tqdm
import time

def sample(N, nodes, phi):
    X, datamodel = sampleFromTorusGraph(
        nodes = nodes,
        samples = N,
        phi = phi,
        fitFCM = False,
        fitPAD = True,
        fitPAS = False,
        return_datamodel = True
    )
    return X, datamodel

def NCE_estimate(X, N, nodes, K):
    X = torch.from_numpy(X).float().T
    noise = torch.rand(N,nodes)*2*torch.tensor(np.pi) 

    model = TorusGraphs(nodes=X.shape[1],K=K)
    model, objective = mixture_torch_loop(X,noise,model)

    theta = model.theta.detach().flatten().numpy()
    return theta

def single_run(cv_runs, N, nodes, phi, K):
    print(f"Running 1 round of {cv_runs} rounds.")
    X, datamodel = sample(N, nodes, phi)
    phiSM, covPhiSM = SM(X, datamodel).compPhiHatAndCovPhiHat()
    theta = NCE_estimate(X, N, nodes, K)

    dist_SM = np.linalg.norm(phi - phiSM)
    dist_NCE = np.linalg.norm(phi - theta)

    return dist_SM, dist_NCE

def cross_val_runs(cv_runs, N, nodes, phi, K, max_workers=None):
    dist_SMs, dist_NCEs = [], []
    i = 1

    with concurrent.futures.ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = [executor.submit(single_run, cv_runs, N, nodes, phi, K) for i in range(cv_runs)]
        results = [future.result() for future in concurrent.futures.as_completed(futures)]

    for dist_SM, dist_NCE in results:
        dist_SMs.append(dist_SM)
        dist_NCEs.append(dist_NCE)

    return dist_SMs, dist_NCEs

    # dist_SMs, dist_NCEs = [], []
    # for i in range(cv_runs):
    #     print(f"\n--- Round {i+1} of {cv_runs}: ---\n")
    #     dist_SM, dist_NCE = single_run(N, nodes, phi, K)
    #     dist_SMs.append(dist_SM)
    #     dist_NCEs.append(dist_NCE)
    # return dist_SMs, dist_NCEs

def plot(dist_SMs, dist_NCEs):
    fig, ax = plt.subplots()
    boxplot_data = [dist_SMs, dist_NCEs]
    ax.boxplot(boxplot_data, patch_artist=True)
    ax.set_xticklabels(['SMs', 'NCEs'])
    ax.set_title('Boxplot of SMs and NCEs')
    plt.show()

if __name__ == "__main__":
    os.environ['DISABLE_TQDM'] = 'True'
    tqdm.__init__ = partialmethod(tqdm.__init__, disable=True)
    start_time = time.time()

    N = 1000 # samples
    nodes = 3
    K = 1 # single model
    cv_runs = 10

    print(f"Estimated time: {10*cv_runs} sec")

    phi = np.block([ 0, 0, 8*np.cos(np.pi), 8*np.sin(np.pi), 0, 0 ])
    dist_SMs, dist_NCEs = cross_val_runs(cv_runs, N, nodes, phi, K)
    print(f"Time taken = {time.time() - start_time}")
    plot(dist_SMs, dist_NCEs)


