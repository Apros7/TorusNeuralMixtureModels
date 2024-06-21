
import sys
sys.path.insert(0, '.')

from src.data.synthetic_data import sampleFromTorusGraph
from src.parameterEstimation.scoreMatching import SM
from src.parameterEstimation.NCE import NCE

import numpy as np
import torch
import matplotlib.pyplot as plt
import concurrent.futures
import os
from functools import partialmethod
import tqdm
import time
from scipy.stats import f_oneway, ttest_rel, wilcoxon, ttest_1samp
import statistics

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

    model = NCE(nodes=X.shape[1],K=K)
    model.run(X,noise)

    theta = model.theta.detach().flatten().numpy()
    return theta

def single_run(cv_runs, N, nodes, phi, K):
    print(f"Running 1 round of {cv_runs} rounds.")
    X, datamodel = sample(N, nodes, phi)
    phiSM, covPhiSM = SM(X, datamodel).compPhiHatAndCovPhiHat()
    theta = NCE_estimate(X, N, nodes, K)

    dist_SM = np.linalg.norm(phi - phiSM)
    dist_NCE = np.linalg.norm(phi - theta)

    dist_SM_abs = np.abs(phi - phiSM)
    dist_NCE_abs = np.abs(phi - theta)


    return dist_SM, dist_NCE, phiSM, theta, dist_SM_abs, dist_NCE_abs

def cross_val_runs(cv_runs, N, nodes, phi, K, max_workers=None):
    dist_SMs, dist_NCEs, phiSMS, thetas, dist_SM_abss, dist_NCE_abss = [], [], [], [], [], []
    i = 1

    with concurrent.futures.ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = [executor.submit(single_run, cv_runs, N, nodes, phi, K) for i in range(cv_runs)]
        results = [future.result() for future in concurrent.futures.as_completed(futures)]

    for dist_SM, dist_NCE, phiSM, theta, dist_SM_abs, dist_NCE_abs in results:
        dist_SMs.append(dist_SM)
        dist_NCEs.append(dist_NCE)
        phiSMS.append(phiSM)
        thetas.append(theta)
        dist_SM_abss.append(dist_SM_abs)
        dist_NCE_abss.append(dist_NCE_abs)


    return dist_SMs, dist_NCEs, phiSMS, thetas, dist_SM_abss, dist_NCE_abss

    # dist_SMs, dist_NCEs = [], []
    # for i in range(cv_runs):
    #     print(f"\n--- Round {i+1} of {cv_runs}: ---\n")
    #     dist_SM, dist_NCE = single_run(N, nodes, phi, K)
    #     dist_SMs.append(dist_SM)
    #     dist_NCEs.append(dist_NCE)
    # return dist_SMs, dist_NCEs

def plot(dist_SMs, dist_NCEs):
    plt.rcParams['font.family'] = 'Times New Roman'
    fig, ax = plt.subplots()
    boxplot_data = [dist_SMs, dist_NCEs]
    ax.boxplot(boxplot_data, patch_artist=True)
    ax.set_xticklabels(['SMs', 'NCEs'])
    ax.set_title('Boxplot of SMs and NCEs distances to true phi')

def plot1(dist_SM_abss, dist_NCE_abss):
    plt.rcParams['font.family'] = 'Times New Roman'
    fig, ax = plt.subplots()
    boxplot_data = [dist_SM_abss[0], dist_NCE_abss[0]]
    ax.boxplot(boxplot_data, patch_artist=True)
    ax.set_xticklabels(['SM', 'NCE'])
    ax.set_title('Boxplot of absolute differences between SM and NCE to true phi')

def plot2(phiSMS, thetas):
    plt.rcParams['font.family'] = 'Times New Roman'
    fig, ax = plt.subplots()
    boxplot_data = [phiSMS[0], thetas[0]]
    ax.boxplot(boxplot_data, patch_artist=True)
    ax.set_xticklabels(['SM', 'NCE'])
    ax.set_title('Boxplot of output of SM and NCE')

if __name__ == "__main__":
    os.environ['DISABLE_TQDM'] = 'True'
    tqdm.__init__ = partialmethod(tqdm.__init__, disable=True)
    start_time = time.time()

    N = 1000 # samples
    nodes = 3
    K = 1 # single model
    cv_runs = 10

    print(f"Estimated time: {10*cv_runs} sec")
    
    plt.rcParams['font.family'] = 'Times New Roman'
    phi = np.block([ 0, 0, 8*np.cos(np.pi), 8*np.sin(np.pi), 0, 0 ])
    dist_SMs, dist_NCEs, phiSMS, thetas, dist_SM_abss, dist_NCE_abss = cross_val_runs(cv_runs, N, nodes, phi, K)
    print(f"Time taken = {time.time() - start_time}")
    plot(dist_SMs, dist_NCEs)
    plt.savefig('src/plots/sm_vs_nce_boxplot.png')
    plt.show()

    plot2(phiSMS, thetas)
    plt.savefig('src/plots/sm_vs_nce_boxplot2.png')
    plt.show()

    plot1(dist_SM_abss, dist_NCE_abss)
    plt.show()

#pvalue for models distance being different from each other
    pvalue_dists = wilcoxon(dist_SMs, dist_NCEs)
    print(f"p-value dists: {pvalue_dists.pvalue}")

#pvalue for models being different from true phi
    pvalue_sm_phi = wilcoxon(np.mean(phiSMS, axis=0), phi)
    print(f"p-value sm_phi: {pvalue_sm_phi.pvalue}")
    if pvalue_sm_phi.pvalue < 0.05:
        print("SM phi is significantly different from true phi\n")
    else:
        print("SM phi is not significantly different from true phi\n")

    pvalue_nce_phi = wilcoxon(np.mean(thetas, axis=0), phi)
    print(f"p-value nce_phi: {pvalue_nce_phi.pvalue}")
    if pvalue_nce_phi.pvalue < 0.05:
        print("NCE phi is significantly different from true phi\n")
    else:
        print("NCE phi is not significantly different from true ph\ni")

#pvalue for models error being different from phi
    pvalue_sm = wilcoxon(phi-np.mean(phiSMS, axis=0), phi)
    print(f"p-value sm: {pvalue_sm.pvalue}")
    if pvalue_sm.pvalue < 0.05:
        print("SM distance to true phi is significantly different from true phi\n")
    else:
            print("SM distance to true phi is not significantly different from true phi\n")

    pvalue_nce = wilcoxon(phi-np.mean(thetas, axis=0),phi)
    print(f"p-value nce: {pvalue_nce.pvalue}")
    if pvalue_nce.pvalue < 0.05:
        print("NCE distance to true phi is significantly different from true phi\n")
    else:
        print("NCE distance to true phi is not significantly different from true phi\n")

#pvalue for models error being different from 0
    pvalue_0_sm = wilcoxon(phi-np.mean(phiSMS, axis=0), phi-phi)
    print(f"p-value sm to 0: {pvalue_0_sm.pvalue}")
    if pvalue_0_sm.pvalue < 0.05:
        print("SM error is significantly different from 0\n")
    else:
        print("SM error is not significantly different from 0\n")

    pvalue_0_nce = wilcoxon(phi-np.mean(thetas, axis=0), phi-phi)
    print(f"p-value nce to 0: {pvalue_0_nce.pvalue}")
    if pvalue_0_nce.pvalue < 0.05:
        print("NCE error is significantly different from 0\n")
    else:
        print("NCE error is not significantly different from 0\n")

#variance for models
    thetas_np = np.array(thetas)
    phiSMS_np = np.array(phiSMS)

    thetas_var = []
    phiSMS_var = []
    for i in range(0,5):
        thetas_var.append(np.var(thetas_np[:,i]))
        phiSMS_var.append(np.var(phiSMS_np[:,i]))

    print(f"Variance SM: {phiSMS_var}")
    print(f"Variance NCE: {thetas_var}")