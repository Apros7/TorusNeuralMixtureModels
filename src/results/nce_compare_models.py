# Do NCE with two different init of theta
import numpy as np
import matplotlib.pyplot as plt
import torch
import sys
sys.path.insert(0, '.')

from src.parameterEstimation.trainNCE import mixture_torch_loop
from src.parameterEstimation.NCE import TorusGraphs
from src.data.synthetic_data import sampleFromTorusGraph

N = 100 # samples
nodes = 3
K = 2 # number of models (or components)
phi1 = np.block([ 0, 0, 8*np.cos(np.pi), 8*np.sin(np.pi), 0, 0 ]) 
X1, datamodel = sampleFromTorusGraph(
        nodes = nodes,
        samples = N,
        phi = phi1,
        fitFCM = False,
        fitPAD = True,
        fitPAS = False,
        return_datamodel = True
    )
X1 = torch.from_numpy(X1).float().T
phi2 = np.block([ -8*np.cos(np.pi), 0, 0, 0, 0, -8*np.sin(np.pi) ]) 
X2, datamodel = sampleFromTorusGraph(
        nodes = nodes,
        samples = N,
        phi = phi1,
        fitFCM = False,
        fitPAD = True,
        fitPAS = False,
        return_datamodel = True
    )
X2 = torch.from_numpy(X2).float().T

X = torch.cat([X1,X2],dim=0)


noise = torch.rand(N*2,nodes)*2*torch.tensor(np.pi) # Noise distribution, mellem 0 og 2*pi


models = TorusGraphs(nodes=X.shape[1],K=2)
models,objective = mixture_torch_loop(X,noise,models)


theta = models.theta
logc = models.logc