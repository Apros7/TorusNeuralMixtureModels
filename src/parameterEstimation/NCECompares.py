import numpy as np
import torch
from tqdm import tqdm
import matplotlib.pyplot as plt

import sys
sys.path.insert(0, '.')

from NCE import TorusGraphs
from src.parameterEstimation.trainNCE import mixture_torch_loop
from src.data.synthetic_data import sampleFromTorusGraph
from src.parameterEstimation.trainNCE import mixture_torch_loop


def plot_nce_vs_nce(X, nodes, K):
    X = torch.from_numpy(X).float().T
    noise = torch.rand(N,nodes)*2*torch.tensor(np.pi)

    #Model 1
    model1 = TorusGraphs(nodes=X.shape[1],K=K)
    model1, objective1 = mixture_torch_loop(X, noise, model1)
    theta1 = model1.theta

    #Model 2    
    model2 = TorusGraphs(nodes=X.shape[1],K=K)
    model2, objective2 = mixture_torch_loop(X, noise, model2)
    theta2 = model2.theta
    

    plt.figure(figsize=(10,7))
    plt.subplot(1,2,1)
    plt.imshow(theta1.detach().numpy()[0,:,:])
    plt.colorbar()
    plt.title("Theta, NCE Model 1")
    plt.xlabel("Dimension")
    plt.ylabel("Model")

    plt.subplot(1,2,2)
    plt.imshow(theta2.detach().numpy()[0,:,:])
    plt.colorbar()
    plt.title("Theta, NCE Model 2")
    plt.xlabel("Dimension")
    plt.ylabel("Model")

    plt.savefig('plot_nce_vs_nce.png')



if __name__ == "__main__":
    N = 100
    nodes = 3
    K = 2
    phi1 = np.block([8*np.cos(np.pi),0,0 ,8*np.sin(np.pi),0,0])
    phi2 = np.block([0, 0, 8*np.cos(np.pi)/2, 8*np.sin(np.pi)/2, 0, 0])
    noise = torch.rand(N,nodes)*2*torch.tensor(np.pi) # Noise distribution
    X, datamodel = sampleFromTorusGraph(
        nodes = nodes, 
        samples = N,
        phi = phi1,
        fitFCM = False,
        fitPAD = True,
        fitPAS = False,
        return_datamodel = True
    )
    X2, datamodel = sampleFromTorusGraph(
        nodes = nodes, 
        samples = N,
        phi = phi2,
        fitFCM = False,
        fitPAD = True,
        fitPAS = False,
        return_datamodel = True
    )
    

    plot_nce_vs_nce(X, nodes, K)
