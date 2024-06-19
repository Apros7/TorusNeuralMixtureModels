import sys
sys.path.insert(0, '.')
from src.toolbox import load_sample_data, TorusGraph, NCE
import time
import numpy as np
import torch
import matplotlib.pyplot as plt
import seaborn as sns
import scipy
import time


def get_alpha_beta(theta):
    '''Alpha and Beta gets computed here with theta as input.
    Alpha and beta will be returned as a symmetric matrix referring to cos_phi and sin_phi. 

    fx theta is torch.shape = (3,2,21)
    
    
    '''
    K = theta.shape[0]
    theta = theta.detach().numpy()
    n = estimation_method.nodes
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




if __name__ == "__main__":

    # os.environ['DISABLE_TQDM'] = 'True'
    start_time = time.time()

    N = 10 # samples
    nodes = 3
    K = 3 
    cv_runs = 1
    nce_steps = 10

    lrs_to_test = [10, 1, 0.5, 0.1, 0.05, 0.01, 0.005, 0.001]
    lr_dict = {lr: [] for lr in lrs_to_test}

    data, true_vals = load_sample_data()
    data = torch.from_numpy(data).float()

    estimation_method = NCE(
        nodes = data.shape[1], 
        K = K, 
        lr = 0.1, 
        steps = nce_steps, 
        return_log_prop_data = True)

    torus_graph = TorusGraph(
        nodes = data.shape[1], 
        samples = data.shape[0], 
        data = data, 
        nModels = K, 
        estimationMethod = estimation_method, 
        true_vals = true_vals)

    #value = torus_graph.evaluate()

    theta = estimation_method.theta
    Ps = get_phi_corr(theta)

    l = np.concatenate((Ps[0].flatten(), Ps[1].flatten(), Ps[2].flatten()))
    max_val = np.max(l)
    min_val = np.min(l)
    print(f'Min: {min}, Max: {max}')

    plt.rcParams['font.family'] = 'Times New Roman'
    plt.figure(figsize=(22,5))
    plt.subplot(1,3,1)
    plot = sns.heatmap(Ps[0], vmin=min_val, vmax=max_val)
    plt.xticks(ticks=np.arange(7) + 0.5, labels=['Visual', 'Somato-Motor','Dorsal Attention','Ventral Attention','Limbic','Control','Default'], rotation=50)
    plt.yticks(ticks=np.arange(7) + 0.5, labels=['Visual', 'Somato-Motor','Dorsal Attention','Ventral Attention','Limbic','Control','Default'], rotation=0)
    plt.title('Component 1')

    plt.subplot(1,3,2)
    plot = sns.heatmap(Ps[1], vmin=min_val, vmax=max_val)
    plt.xticks(ticks=np.arange(7) + 0.5, labels=['Visual', 'Somato-Motor','Dorsal Attention','Ventral Attention','Limbic','Control','Default'], rotation=50)
    plt.yticks(ticks=np.arange(7) + 0.5, labels=['Visual', 'Somato-Motor','Dorsal Attention','Ventral Attention','Limbic','Control','Default'], rotation=0)
    plt.title('Component 2')

    plt.subplot(1,3,3)
    plot = sns.heatmap(Ps[2], vmin=min_val, vmax=max_val)
    plt.xticks(ticks=np.arange(7) + 0.5, labels=['Visual', 'Somato-Motor','Dorsal Attention','Ventral Attention','Limbic','Control','Default'], rotation=50)
    plt.yticks(ticks=np.arange(7) + 0.5, labels=['Visual', 'Somato-Motor','Dorsal Attention','Ventral Attention','Limbic','Control','Default'], rotation=0)
    plt.title('Component 3')
    plt.savefig('src/plots/real_data_phi_heatmap.png')
    plt.show()

    
