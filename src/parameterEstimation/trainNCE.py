import numpy as np
import torch
from tqdm import tqdm
import os

import sys
sys.path.insert(0, '.')
from src.data.synthetic_data import sampleFromTorusGraph
from src.parameterEstimation.NCE import NCE



def mixture_torch_loop(X,noise,model, lr=0.1):


    optimizer = torch.optim.Adam(model.parameters(),lr=lr)
    objective = []

    for epoch in tqdm(range(model.nce_steps), desc="NCE training", disable=os.environ.get("DISABLE_TQDM", False)):
        if model.return_log_prop_data:    
            obj, log_prop_data = model.NCE_objective_function(X,noise)
            obj = -obj
        else:
            obj = -model.NCE_objective_function(X,noise)


        if torch.isnan(-obj):
            raise ValueError("Nan reached")
        
        optimizer.zero_grad(set_to_none=True)
        obj.backward()
        optimizer.step()
        objective.append(-obj.item())
            
    if model.return_log_prop_data:
        return model,objective,log_prop_data
    return model,objective


if __name__=="__main__":
   N=100 # samples
   nodes=3 

   X = sampleFromTorusGraph(nodes,N,fitFCM=False,fitPAD=True,fitPAS=False)
   X = torch.from_numpy(X).float().T
   noise = torch.rand(N,nodes)*2*torch.tensor(np.pi) # Noise distribution


   model = NCE(nodes=X.shape[1],K=1)
   model,objective = mixture_torch_loop(X,noise,model)


   theta = model.theta
   logc = model.logc


   import matplotlib.pyplot as plt
   plt.figure(figsize=(10,6))
#    plt.subplot(1,2,1)
   plt.plot(objective)
   plt.title(f"Objective function NCE with {model.K} models")
   plt.xlabel("Iterations")
   plt.ylabel("Loss")
   
#    plt.subplot(1,2,2)
#    plt.imshow(theta.detach().numpy()[0,:,:])
#    plt.colorbar()
#    plt.title("Theta")
#    plt.xlabel("Dimension")
#    plt.ylabel("Model")

   plt.show()
   
   # plt.subplot(1,3,3)
   # plt.plot(logc.detach().numpy())
   # plt.title("Logc")
   # plt.xlabel("Model")
   # plt.ylabel("Logc")

   # plt.savefig('tmp.png')
   # print(theta,logc)



## lav sammenlign mellem 2 modeller med forskellige init_theta