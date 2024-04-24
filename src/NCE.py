## torch class

import torch as torch
import numpy as np
import math
from torch import logsumexp

# en init funktion containing all parameters for NCE

class NCE(torch.nn.Module):

    def __init__(self, c, ptilde, phi):
        super(NCE, self).__init__()
        self.c = c
        self.ptilde = ptilde
        self.phi = phi
        N = len(self.phi)
        M = len(self.c)

    def log_p_TG(self, ptilde, c): # This is the log exp for the single torus graph
        # 1. Implement the logsumexp function. This below is probably not right. 
        return torch.log(torch.exp(ptilde + c)) 
    
    def log_p_TGMM(self, ptilde, c): # This is the log exp for the multible torus graph hence a sum.
        return logsumexp(ptilde + c) 

    def loss(self, phi, c, y, x, theta,p,n): 
        # n(x),n(y) is a noise function that is not implemented yet.
        # p(x, theta, c) is the model probability distribution which is either log_p_TG or log_p_TGMM
        if n == None:
            n = 1

        if p == 'TG':
            N = len(phi)
            M = len(c)
            term1 = torch.sum(torch.log(N * p(x, theta, c) / (N * p(x, theta, c) + M*n(x))) / N)
            term2 = torch.sum(torch.log(M*n(y) / (N * p(y, theta, c) + M*n(y))) / N)
            return term1 + term2
        elif p == 'TGMM':
            N = len(phi)
            M = len(c)
            term1 = torch.sum(torch.log(N * p(x, theta, c) / (N * p(x, theta, c) + M*n(x))) / N)
            term2 = torch.sum(torch.log(M*n(y) / (N * p(y, theta, c) + M*n(y))) / N)
        else:
            raise ValueError('probability distribution must be either TG or TGMM')
        return term1 + term2

        
