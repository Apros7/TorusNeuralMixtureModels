# Do NCE with two different init of theta
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import torch
import sys
sys.path.insert(0, '.')

from src.parameterEstimation.trainNCE import mixture_torch_loop
from src.parameterEstimation.NCE import TorusGraphs
from src.data.synthetic_data import sampleFromTorusGraph
from src.results.NMI import calc_NMI, calc_MI

N = 200 # samples
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
phi2 = np.block([ 0, 0, 8*np.sin(np.pi), 8*np.cos(np.pi), 0, 0 ]) 
X2, datamodel = sampleFromTorusGraph(
        nodes = nodes,
        samples = N,
        phi = phi2,
        fitFCM = False,
        fitPAD = True,
        fitPAS = False,
        return_datamodel = True
    )
X2 = torch.from_numpy(X2).float().T

X = torch.cat([X1,X2],dim=0)

N = 2*N
noise = torch.rand(N,nodes)*2*torch.tensor(np.pi) # Noise distribution, mellem 0 og 2*pi


# model1 = TorusGraphs(nodes=X1.shape[1],K=1,return_log_prop_data=True)
# model1,objective1, log_prob_data1 = mixture_torch_loop(X1,noise,model1)
# theta1,c1 = model1.theta,model1.logc

# model2 = TorusGraphs(nodes=X2.shape[1],K=1,return_log_prop_data=True)
# model2,objective2, log_prob_data2 = mixture_torch_loop(X2,noise,model2)
# theta2,c2 = model2.theta,model2.logc


model = TorusGraphs(nodes=X.shape[1],K=2,return_log_prop_data=True)
model,objective, log_prob_data = mixture_torch_loop(X,noise,model)
theta,c = model.theta,model.logc
print(log_prob_data.shape) # (K,N)
log_prob_data = torch.tensor(log_prob_data)


def compute_log_probs(self, log_prob_data, c):
    denominator = torch.sum(torch.exp(log_prob_data + self.logc.view(-1,1)), dim=0)
    z = torch.zeros(self.K, N)
    for k in range(self.K):
        for i in range(N):
            z[k, i] = torch.exp(log_prob_data[k, i] + c[k]) / denominator[i]
    return z


def classify_points(model, X, log_prob_data, c):
    log_probs = compute_log_probs(model, log_prob_data, c)
    print(log_probs.size())
    class_label = torch.argmax(log_probs, dim=0)
    return class_label

label = classify_points(model, X, log_prob_data, c)
# plot label dsitrubtion of the data
ones = torch.sum(label == 1).item()
wrong_ones = abs(N/2 - ones)
zeros = torch.sum(label == 0).item()
wrong_zeros = wrong_ones

# Confusion matrix
confusion_matrix = torch.zeros(2,2)
confusion_matrix[0,0] = ones
confusion_matrix[0,1] = wrong_zeros
confusion_matrix[1,0] = wrong_ones
confusion_matrix[1,1] = zeros

# plot 
sns.heatmap(confusion_matrix, annot=confusion_matrix, annot_kws={"size": 8})
plt.title("Mixture model NCE with two components, class label distribution")
plt.show()
plt.savefig("src/plots/mix_nce.png")

# NMI: Normalized Mutual Information
z = compute_log_probs(model, log_prob_data, c)
z1,z2 = torch.tensor_split(z,2,dim=1)
z1 = z1.detach().numpy()
z2 = z2.detach().numpy()

MI = calc_MI(z1,z2)
NMI = calc_NMI(z1,z2)

# Plot boxplot of NMI and MI
plt.boxplot([[NMI,0],[MI,0]])
plt.xticks([1,2],["NMI","MI"])
plt.title("NMI and MI for two components")
plt.show()
plt.savefig("src/plots/NMI.png")