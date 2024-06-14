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

# z : prob of point x belonging to model k
# z = torch.exp(log_prop_data_1 + logc)/torch.sum(torch.exp(log_prop_data + logc),dim=0)
def compute_log_probs(self, X):
    N = X.shape[0]
    log_prob_data = torch.zeros(self.K, N)
    for k in range(self.K):
        for i in range(N):
            for z in range(self.nodes*(self.nodes-1)//2):
                cosx = torch.cos(X[i, self.triu_indices[0, z]] - X[i, self.triu_indices[1, z]])
                sinx = torch.sin(X[i, self.triu_indices[0, z]] - X[i, self.triu_indices[1, z]])
                log_prob_data[k, i] += torch.sum(self.theta[k, 0, z]*cosx + self.theta[k, 1, z]*sinx)

    z = torch.exp(log_prob_data + self.logc) / torch.sum(torch.exp(log_prob_data + self.logc), dim=0)
    return z


def classify_points(X):
    log_probs = compute_log_probs(models, X)
    class_labels = torch.argmax(log_probs, dim=1)
    return class_labels

labels = classify_points(X)
# plot label dsitrubtion of the data
plt.hist(labels.numpy())
plt.show()