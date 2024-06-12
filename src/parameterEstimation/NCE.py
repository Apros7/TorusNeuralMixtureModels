
import torch
import torch.nn as nn


class TorusGraphs(nn.Module):
   def __init__(self, nodes:int, K:int):
       '''
       nodes: number of nodes
       K: number of models (or components) 
       
       '''
       super().__init__()


       self.K = K
       self.nodes = nodes
       z = nodes*(nodes-1)//2
       self.K = 1
       self.theta = nn.Parameter(torch.randn(self.K,2,z))
       self.logc = nn.Parameter(torch.zeros(self.K))


       self.triu_indices = torch.triu_indices(nodes,nodes,offset=1)


   def NCE_objective_function(self,X,noise):
       
       """This function computes the NCE objective function for an arbitrary number of the Torus Graphs models.
       Thus it works for mixture models as well.
       Args:
         X: torch.tensor of shape (N,nodes) where N is the number of samples and nodes is the number of nodes
    noise: torch.tensor of shape (N,nodes) where N is the number of samples and nodes is the number of nodes

         Returns:
            J: torch.tensor of shape (K) where K is the number of models
            If you do:  

            model = TorusGraphs(nodes=X.shape[1],K=1)
            model,objective = mixture_torch_loop(X,noise,model)

            Then you can get theta and c by doing:
            theta = model.theta
            logc = model.logc

    
       
    """

       # theta (Mx2xz)
       # x (Nxp)
       N = X.shape[0]
       M = noise.shape[0] #number of noise samples


       log_prob_data = torch.zeros(self.K,N)
       log_prob_noise = torch.zeros(self.K,N)
    #    for k in range(self.K):
    #        for i in range(N):
    #            for z in range(self.p*(self.p-1)//2):
    #                cosx = torch.cos(X[i,self.triu_indices[0,z]] - X[i,self.triu_indices[1,z]])
    #                sinx = torch.sin(X[i,self.triu_indices[0,z]] - X[i,self.triu_indices[1,z]])
    #                log_prob_data[k,i] += self.theta[k,:,z]@torch.tensor([cosx,sinx])


    #                cosn = torch.cos(noise[i,self.triu_indices[0,z]] - noise[i,self.triu_indices[1,z]])
    #                sinn = torch.sin(noise[i,self.triu_indices[0,z]] - noise[i,self.triu_indices[1,z]])
    #                log_prob_noise[k,i] += self.theta[k,:,z]@torch.tensor([cosn,sinn])
       

       for z in range(self.nodes*(self.nodes-1)//2):
           cosx = torch.cos(X[:,self.triu_indices[0,z]] - X[:,self.triu_indices[1,z]])
           sinx = torch.sin(X[:,self.triu_indices[0,z]] - X[:,self.triu_indices[1,z]])
           log_prob_data = self.theta[:,:,z]@torch.stack([cosx,sinx],dim=0)


           cosn = torch.cos(noise[:,self.triu_indices[0,z]] - noise[:,self.triu_indices[1,z]])
           sinn = torch.sin(noise[:,self.triu_indices[0,z]] - noise[:,self.triu_indices[1,z]])
           log_prob_noise = self.theta[:,:,z]@torch.stack([cosn,sinn],dim=0)
       log_prob_data = torch.logsumexp(log_prob_data + self.logc,dim=0)
       log_prob_noise = torch.logsumexp(log_prob_noise + self.logc,dim=0)


       log_nx = torch.zeros(N) #needs to be implemented, this is the noise samples
       log_ny = torch.zeros(N) # maybe just noise


       log_J1_denom = torch.logsumexp(torch.stack([torch.log(torch.tensor(N))+log_prob_data,torch.log(torch.tensor(M))+log_nx],dim=-1), dim=-1)
       log_J2_denom = torch.logsumexp(torch.stack([torch.log(torch.tensor(N))+log_prob_noise,torch.log(torch.tensor(M))+log_ny],dim=-1), dim=-1)


       J = torch.mean(torch.log(torch.tensor(N))+log_prob_data - log_J1_denom,dim=-1)+torch.mean(torch.log(torch.tensor(N))+log_ny - log_J2_denom,dim=-1)

    

       return J
