import sys
sys.path.insert(0, '.')

from src.toolbox import TorusGraph, NCE, SM, sample_syndata_torusgraph, TorusGraphInformation
from src.parameterEstimation.trainNCE import mixture_torch_loop


import time
import os
import torch
import matplotlib.pyplot as plt
import numpy as np
from typing import Tuple
import concurrent.futures
from functools import partialmethod
import tqdm
from scipy.stats import f_oneway
from scipy import stats


def sample_data(N, nodes, phi, nModels ) -> Tuple[np.ndarray, TorusGraphInformation]:
    X, datamodel = sample_syndata_torusgraph(
        nodes = nodes,
        samples = N,
        phi = phi,
        nModels = nModels,
        fitFCM = False,
        fitPAD = True,
        fitPAS = False,
        return_datamodel = True
    )
    return X, datamodel

def NCE_estimate(X, N, nodes, K, lr = 0.1, nce_steps = 2000):
    X = torch.from_numpy(X).float().T
    noise = torch.rand(N,nodes)*2*torch.tensor(np.pi)
    estimation_method = NCE(nodes = X.shape[1], K = K, lr = lr, steps = nce_steps, return_log_prop_data = True)
    torus_graph = TorusGraph(nodes = X.shape[1], samples = X.shape[0], data = X, nModels = K, estimationMethod = estimation_method)

    theta = estimation_method.theta.detach().flatten().numpy()

    return theta

def single_run(N, nodes, phi, K):
    X, datamodel = sample_data(N, nodes, phi, nModels =2)
    theta = NCE_estimate(X, N, nodes, K)

    return theta

def cross_val_runs(cv_runs, N, nodes, phi, K):
    thetas = []

    for i in range(cv_runs):
        print(f"\n--- Round {i+1} of {cv_runs}: ---\n")
        # dist, theta = run_NCE(N, nodes, phi, K)
        # dists.append(dist)
        theta = single_run(N, nodes, phi, K)
        thetas.append(theta)

    thetas = np.array(thetas)
    return thetas

def distances(thetas, cv_runs):
    dists = []

    for i in range(cv_runs):
        dist = np.array(np.linalg.norm(phi[0] - thetas[0,i].reshape(2,6)))
        dists.append(dist)
        dist1 = np.array(np.linalg.norm(phi[1] - thetas[1,i].reshape(2,6)))
        dists.append(dist1)
        
    return np.array(dists)

def plots(dists):
    dist1 = dists[:,0,0]#.reshape(20,1)
    dist2 = dists[:,0,1]#.reshape(20,1)
    dist3 = dists[:,0,2]#.reshape(1,20)
    dist4 = dists[:,0,3]#.reshape(1,20)
    dist5 = dists[:,0,4]#.reshape(1,20)
    dist6 = dists[:,0,5]#.reshape(1,20)

    plt.rcParams['font.family'] = 'Times New Roman'
    fig, ax = plt.subplots()
    boxplot_data = [dist1, dist2, dist3, dist4, dist5, dist6] #, dist3, dist4, dist5, dist6]
    #boxplot_data = dist1
    ax.boxplot(boxplot_data, patch_artist=True)
    ax.set_xticklabels(['Phi1', 'Phi2', 'Phi3', 'Phi4', 'Phi5', 'Phi6'])
    ax.set_title('Boxplot of distances to true Phi values')
    plt.show()


def stat_tests(thetas):
    pvalues = []
    mean = []
    std = []
    model_pairs = []

    for i in range(len(thetas)):
        mean.append(np.mean(thetas[:,i]))
        std.append(np.std(thetas[:,i]))
        for j in range(i + 1, len(thetas)):  # Ensure each pair is tested once
            pvalue = stats.f_oneway(thetas[i], thetas[j])
            pvalues.append(pvalue.pvalue)
            model_pairs.append((i, j))
        # for j in range(len(thetas)):
        #     pvalue = stats.f_oneway(thetas[i], thetas[j])
        #     pvalues.append(pvalue.pvalue)
        #     if j == len(thetas):
        #         pvalue = f_oneway(thetas[-1], thetas[0])
        #         pvalues.append(pvalue.pvalue)

    return mean, std, pvalues, model_pairs



if __name__=="__main__":
    os.environ['DISABLE_TQDM'] = 'True'
    tqdm.__init__ = partialmethod(tqdm.__init__, disable=True)
    start_time = time.time()

    N = 100
    nodes = 3
    K = 2
    cv_runs = 10
    phi = [
        np.block([ 0, 0, 8*np.cos(np.pi), 8*np.sin(np.pi), 0, 0 ]), 
        np.block([ 0, 0, 8*np.cos(np.pi/2), 8*np.sin(np.pi/2), 0, 0 ])]

    thetas = cross_val_runs(cv_runs, N, nodes, phi, K)
    dists = distances(thetas, cv_runs)
    mean, std, pvalues, model_pairs = stat_tests(thetas)
    plots(dists)

    #plots(dist1, dist2, dist3, dist4, dist5, dist6)

    #print(pvalues)
    significant_pairs = [(pair, pval) for pair, pval in zip(model_pairs, pvalues) if pval < 0.05]

    for pair, pval in significant_pairs:    
        print(f"Model {pair[0]} and Model {pair[1]} are significantly different with p-value: {pval}")

    plt.figure(figsize=(10,10))
    plt.subplot(2,3,1)
    plt.plot(range(cv_runs), thetas[:,0])
    plt.axhline(mean[0], color='blue', linestyle='--', linewidth=2, label='Mean')
    plt.xlabel('Iterations')
    plt.ylabel('Theta values')
    plt.title('Phi1')
    plt.legend()

    plt.subplot(2,3,2)
    plt.plot(range(cv_runs), thetas[:,1], color='green')
    plt.axhline(mean[1], color='green', linestyle='--', linewidth=2, label='Mean')
    plt.xlabel('Iterations')
    plt.ylabel('Theta values')
    plt.title('Phi2')
    plt.legend()

    plt.subplot(2,3,3)
    plt.plot(range(cv_runs), thetas[:,2], color='red')
    plt.axhline(mean[2], color='red', linestyle='--', linewidth=2, label='Mean')
    plt.xlabel('Iterations')
    plt.ylabel('Theta values')
    plt.title('Phi3')
    plt.legend()

    plt.subplot(2,3,4)
    plt.plot(range(cv_runs), thetas[:,3], color='purple')
    plt.axhline(mean[3], color='purple', linestyle='--', linewidth=2, label='Mean')
    plt.xlabel('Iterations')
    plt.ylabel('Theta values')
    plt.title('Phi4')
    plt.legend()

    plt.subplot(2,3,5)
    plt.plot(range(cv_runs), thetas[:,3], color='orange')
    plt.axhline(mean[4], color='orange', linestyle='--', linewidth=2, label='Mean')
    plt.xlabel('Iterations')
    plt.ylabel('Theta values')
    plt.title('Phi5')
    plt.legend()

    plt.subplot(2,3,6)
    plt.plot(range(cv_runs), thetas[:,5], color='teal')
    plt.axhline(mean[5], color='teal', linestyle='--', linewidth=2, label='Mean')
    plt.xlabel('Iterations')
    plt.ylabel('Theta values')
    plt.title('Phi6')
    plt.legend()
    plt.show()