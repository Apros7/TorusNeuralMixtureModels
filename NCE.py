## torch class

import torch
import numpy 
import math

# en init funktion containing all parameters for NCE

def NCE( torch.nn.Module ):

    def __init__(self, c, ptilde, phi):
        super(NCE, self).__init__()
        self.c = c
        self.ptilde = ptilde
        self.phi = phi

    def log_p_TG(self, ptilde, c):
        return torch.logexp(ptilde + c)
    
    def log_p_TGMM(self, ptilde, c):
        return torch.logsumexp(ptilde + c) 

    def loss(self, phi, c, noise):
        raise('implement loss function here')
        
