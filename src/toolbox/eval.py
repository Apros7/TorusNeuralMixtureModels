
import numpy as np
import torch

def visualize_phase_coherens(): pass

def calc_MI(Z1,Z2):
   P=Z1@Z2.T
   PXY=P/np.sum(P)
   PXPY=np.outer(np.sum(PXY,axis=1),np.sum(PXY,axis=0))
   ind=np.where(PXY>0)
   MI=np.sum(PXY[ind]*np.log(PXY[ind]/PXPY[ind]))
   return MI


def calc_NMI(Z1,Z2):
   Z1 = np.double(Z1)
   Z2 = np.double(Z2)
   #Z1 and Z2 are two partition matrices of size (KxN) where K is number of components and N is number of samples
   NMI = (2*calc_MI(Z1,Z2))/(calc_MI(Z1,Z1)+calc_MI(Z2,Z2))
   return NMI


def compute_log_probs(estimationMethod, log_prob_data, samples):
    denominator = torch.sum(torch.exp(log_prob_data + estimationMethod.logc.view(-1,1)), dim=0)
    z = torch.zeros(estimationMethod.K, samples)
    for k in range(estimationMethod.K):
        for i in range(samples):
            z[k, i] = torch.exp(log_prob_data[k, i] + estimationMethod.logc[k]) / denominator[i]
    return z


def classify_points(estimationMethod, log_prob_data, N):
    log_prob_data = torch.tensor(log_prob_data)
    log_probs = compute_log_probs(estimationMethod, log_prob_data, N)
    class_label = torch.argmax(log_probs, dim=0)
    return class_label

def adjust_pred_labels(pred_labels, K):
    # Used to adjust pred labels so it matches formation 
    # The torus graphs may think the first true label are mostly ones, which gives a bad score
    # although it succeeded in splitting the clusters
    pass


def ohe(tensor, K): return torch.nn.functional.one_hot(tensor, num_classes=K) # one hot encoding