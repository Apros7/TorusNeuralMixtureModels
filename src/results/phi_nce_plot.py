import sys

import scipy.special
sys.path.insert(0, '.')

from src.toolbox import load_sample_data, TorusGraph, NCE, sample_syndata_torusgraph
import time
import numpy as np
import torch
import matplotlib.pyplot as plt
import seaborn as sns
import os
import scipy
import time
from src.data.synthetic_data import sampleFromTorusGraph
from src.parameterEstimation.NCE import NCE
from src.parameterEstimation.trainNCE import mixture_torch_loop


def get_alpha_beta(theta):
    '''Alpha and Beta gets computed here with theta as input.
    Alpha and beta will be returned as a symmetric matrix referring to cos_phi and sin_phi. 

    fx theta is torch.shape = (3,2,21)
    
    
    '''
    K = model.K
    theta = theta.detach().numpy()
    n = nodes
    idx = torch.triu_indices(n,n,1)
    alpha = [np.zeros((n,n)) for i in range(K)]
    beta = alpha
    for component in np.arange(0,K):
        for j in np.arange(0,idx.shape[1]):
            alpha[component][idx[0,j],idx[1,j]] = theta[component][0][j]
            beta[component][idx[0,j],idx[1,j]] = theta[component][1][j]

        alpha[component] += alpha[component].T  
        beta[component] += beta[component].T    
        np.fill_diagonal(alpha[component], 1) 
        np.fill_diagonal(beta[component], 1) 


    return alpha, beta

def I0(x):
    # scipy.special.i0(x) # Bessel function of the first kind of real order and complex argument
    # x = input, alpha * beta .....
    return scipy.special.i0(x)
def I1(x):
    # scipy.special.i1(x) # Bessel function of the first kind of real order and complex argument
    return scipy.special.i1(x)

def get_phi_corr(theta): # this returns P_jk, but should be a matrix of correlations between signals
    alpha, beta = get_alpha_beta(theta)
    n = alpha[0].shape[0]
    P = [np.zeros((n,n)) for i in range(K)]
    for component in range(K):
        for row in range(n):
            for col in range(n):
                P[component][row,col] = I1(np.sqrt(alpha[component][row,col]**2 + beta[component][row,col]**2)) / I0(np.sqrt(alpha[component][row,col]**2 + beta[component][row,col]**2))
    return P


def get_true_phi(phi):
    K = model.K
    n = nodes
    idx = torch.triu_indices(n,n,1)
    alpha = [np.zeros((n,n)) for i in range(K)]
    beta = alpha
    Phi = []
    for component in np.arange(0,K):
        Phi.append(phi[component].reshape(-1,K))
        for j in np.arange(0,idx.shape[1]):
            alpha[component][idx[0,j],idx[1,j]] = Phi[component][0][j]
            beta[component][idx[0,j],idx[1,j]] = Phi[component][1][j]

        alpha[component] += alpha[component].T  
        beta[component] += beta[component].T    
        np.fill_diagonal(alpha[component], 1) 
        np.fill_diagonal(beta[component], 1) 
    n = alpha[0].shape[0]
    P = [np.zeros((n,n)) for i in range(K)]
    for component in range(K):
        for row in range(n):
            for col in range(n):
                P[component][row,col] = I1(np.sqrt(alpha[component][row,col]**2 + beta[component][row,col]**2)) / I0(np.sqrt(alpha[component][row,col]**2 + beta[component][row,col]**2))
    return P


if __name__ == "__main__":

    # os.environ['DISABLE_TQDM'] = 'True'
    start_time = time.time()

    N = 1000 # samples
    nodes = 3
    K = 3 
    cv_runs = 10
    nce_steps = 5000

    lrs_to_test = [10, 1, 0.5, 0.1, 0.05, 0.01, 0.005, 0.001]
    lr_dict = {lr: [] for lr in lrs_to_test}

    N = 200 # samples
    nodes = 3
    K = 3 # number of models (or components)

    data, datamodel = sampleFromTorusGraph(
            nodes = nodes,
            samples = N,
            phi = None,
            fitFCM = False,
            fitPAD = True,
            fitPAS = False,
            return_datamodel = True
        )
    data = torch.from_numpy(data).float().T
    noise = torch.rand(N,nodes)*2*torch.tensor(np.pi) # Noise distribution, mellem 0 og 2*pi


    model = NCE(nodes=data.shape[1],K=3,return_log_prop_data=False)
    model,objective = mixture_torch_loop(data,noise,model)
    theta,c = model.theta,model.logc

    Ps = get_phi_corr(theta)

    plt.figure(figsize=(16,4))
    plt.subplot(1,3,1)
    plot = sns.heatmap(Ps[0])
    plt.title('Component 1')

    plt.subplot(1,3,2)
    plot = sns.heatmap(Ps[1])
    plt.title('Component 2')

    plt.subplot(1,3,3)
    plot = sns.heatmap(Ps[2])
    plt.title('Component 3')

    plt.savefig('src/plots/syn_data_phi_heatmap.png')
    plt.show()

    # Doing it for the true phi:
    phi = [
        np.block([ 0, 0, 8*np.cos(np.pi), 8*np.sin(np.pi), 0, 0 ]), 
        np.block([ 0, 0, 8*np.sin(np.pi), 8*np.cos(np.pi), 0, 0 ]), 
        np.block([ 0, 0, 8*np.cos(np.pi)*np.cos(np.pi), 8*np.cos(np.pi)*np.sin(np.pi), 0, 0 ])
    ]
    true_phis = get_true_phi(phi)

    plt.figure(figsize=(16,4))
    plt.subplot(1,3,1)
    plot = sns.heatmap(true_phis[0])
    plt.title('Component 1')

    plt.subplot(1,3,2)
    plot = sns.heatmap(true_phis[1])
    plt.title('Component 2')

    plt.subplot(1,3,3)
    plot = sns.heatmap(true_phis[2])
    plt.title('Component 3')

    plt.savefig('src/plots/true_phi.png')
    plt.show()
    
