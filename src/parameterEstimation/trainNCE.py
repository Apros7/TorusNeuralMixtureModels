import numpy as np
import torch
from tqdm import tqdm
from NCE import TorusGraphs
import sys
sys.path.insert(0, '.')
from src.data.synthetic_data import sampleFromTorusGraph



def mixture_torch_loop(X,noise,model):


    optimizer = torch.optim.Adam(model.parameters(),lr=0.4)
    objective = []


    for epoch in tqdm(range(2000)):
            
        obj = -model.NCE_objective_function(X,noise) 


        if torch.isnan(-obj):
            raise ValueError("Nan reached")
        
        optimizer.zero_grad(set_to_none=True)
        obj.backward()
        optimizer.step()
        objective.append(-obj.item())
            
    return model,objective


if __name__=="__main__":
   N=100 # samples
   p=3 # nodes

   X = sampleFromTorusGraph(p,N,fitFCM=False,fitPAD=True,fitPAS=False)
   X = torch.from_numpy(X).float().T
   noise = torch.rand(N,p)*2*torch.tensor(np.pi) # Noise distribution


   model = TorusGraphs(p=X.shape[1],K=1)
   model,objective = mixture_torch_loop(X,noise,model)


   theta = model.theta
   logc = model.logc


   import matplotlib.pyplot as plt
   plt.figure(figsize=(10,10))
   plt.subplot(1,2,1)
   plt.plot(objective)
   plt.title(f"Objective function NCE with {model.K} models")
   plt.xlabel("Iterations")
   plt.ylabel("Loss")
   
   plt.subplot(1,2,2)
   plt.imshow(theta.detach().numpy()[0,:,:])
   plt.colorbar()
   plt.title("Theta")
   plt.xlabel("Dimension")
   plt.ylabel("Model")
   
   # plt.subplot(1,3,3)
   # plt.plot(logc.detach().numpy())
   # plt.title("Logc")
   # plt.xlabel("Model")
   # plt.ylabel("Logc")

   plt.savefig('tmp.png')
   print(theta,logc)



## lav sammenlign mellem 2 modeller med forskellige init_theta