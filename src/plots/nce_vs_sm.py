import sys
import matplotlib.pyplot as plt
import numpy as np
import torch
import seaborn as sns  

sys.path.insert(0, '.')

from src.parameterEstimation.trainNCE import mixture_torch_loop
from src.parameterEstimation.NCE import TorusGraphs
from src.data.synthetic_data import sampleFromTorusGraph
from src.parameterEstimation.scoreMatching import SM


def plot_nce(X, nodes, K):
   X = torch.from_numpy(X).float().T
   noise = torch.rand(N,nodes)*2*torch.tensor(np.pi) 

   model = TorusGraphs(nodes=X.shape[1],K=K)
   model,objective = mixture_torch_loop(X,noise,model)

   theta = model.theta
   # logc = model.logc
   
   plt.figure(figsize=(10,10))
   plt.subplot(1,2,1)
   plt.plot(objective)
   plt.title(f"Objective function NCE with {model.K} models")
   plt.xlabel("Iterations")
   plt.ylabel("Loss")

   plt.subplot(1,2,2)
   plt.imshow(theta.detach().numpy()[0,:,:])
   plt.colorbar()
   plt.title("Theta, NCE")
   plt.xlabel("Dimension")
   plt.ylabel("Model")

   plt.savefig('plot_nce.png')
   plt.show()


def plot_sm(X, datamodel):
   phi, covPhi = SM(X, datamodel).compPhiHatAndCovPhiHat()
   plt.imshow(phi)
   plt.colorbar()
   plt.title("Theta, score matching")
   plt.xlabel("Dimension")
   plt.ylabel("Model")

   plt.savefig('plot_sm.png')
   plt.show()

  




if __name__ == "__main__":
   N = 100 # samples
   nodes = 3
   K = 1 # number of models (or components)
   phi = np.block([ 0, 0, 8*np.cos(np.pi), 8*np.sin(np.pi), 0, 0 ])
   X, datamodel = sampleFromTorusGraph(
         nodes = nodes,
         samples = N,
         phi = phi,
         fitFCM = False,
         fitPAD = True,
         fitPAS = False,
         return_datamodel = True
      )

   plot_nce(X, nodes, K)
   plot_sm(X, datamodel)
