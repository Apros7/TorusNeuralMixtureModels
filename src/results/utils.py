
import sys
sys.path.insert(0, '.')

from src.data.synthetic_data import sampleFromTorusGraph
from src.parameterEstimation.trainNCE import mixture_torch_loop
from src.parameterEstimation.NCE import TorusGraphs

import numpy as np
import torch
import matplotlib.pyplot as plt
import concurrent.futures
import os
from functools import partialmethod
from tqdm import tqdm
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

def NCE_estimate(X, N, nodes, K, lr = 0.1):
    X = torch.from_numpy(X).float().T
    noise = torch.rand(N,nodes)*2*torch.tensor(np.pi) 

    model = TorusGraphs(nodes=X.shape[1],K=K)
    model, objective = mixture_torch_loop(X,noise,model, lr= lr)

    theta = model.theta.detach().flatten().numpy()
    return lr, theta

