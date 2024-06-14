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


noise = torch.rand(N,nodes)*2*torch.tensor(np.pi) # Noise distribution, mellem 0 og 2*pi


# model1 = TorusGraphs(nodes=X1.shape[1],K=1,return_log_prop_data=True)
# model1,objective1, log_prob_data1 = mixture_torch_loop(X1,noise,model1)
# theta1,c1 = model1.theta,model1.logc

# model2 = TorusGraphs(nodes=X2.shape[1],K=1,return_log_prop_data=True)
# model2,objective2, log_prob_data2 = mixture_torch_loop(X2,noise,model2)
# theta2,c2 = model2.theta,model2.logc


# Classify the log_prop_data
model2 = TorusGraphs(nodes=X1.shape[1],K=2,return_log_prop_data=True)
model2,objective2, log_prob_data2 = mixture_torch_loop(X2,noise,model2)
theta2,c2 = model2.theta,model2.logc
print(log_prob_data2.shape) # (K,N)