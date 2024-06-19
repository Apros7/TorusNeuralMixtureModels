import sys
import matplotlib.pyplot as plt
import numpy as np
import torch
import seaborn as sns  
import logging

sys.path.insert(0, '.')

# from src.parameterEstimation.trainNCE import mixture_torch_loop
# from src.parameterEstimation.NCE import NCE
# from src.data.synthetic_data import sampleFromTorusGraph
# from src.parameterEstimation.scoreMatching import SM
from src.toolbox import TorusGraph, sample_syndata_torusgraph, NCE, SM

  
def plot_nce_vs_sm(X,nodes, K, datamodel, N):
   sm_tg = TorusGraph(
      nodes = nodes,
      data = X,
      samples = N,
      nModels = K,
      TGInformation = datamodel,
      return_datamodel=True,
      estimationMethod=SM(
         X = X,
         dm = datamodel
      )
   )
   phi = sm_tg.estimationMethod.phi

   nce = NCE(
      nodes = nodes,
      K = K,
      lr = 1,
      steps = 500
   )
   # nce_tg = TorusGraph(
   #    nodes = nodes,
   #    data = X,
   #    samples = N,
   #    nModels = K,
   #    TGInformation = datamodel,
   #    estimationMethod = nce
   # )
   theta = nce.theta.detach().numpy()

   min_val = min(theta.min(), phi.min())
   max_val = max(theta.max(), phi.max())
   
   plt.rcParams['font.family'] = 'Times New Roman'
   plt.figure(figsize=(10,7))

   plt.subplot(1,2,1)
   plt.imshow(theta[0,:,:], vmin=min_val, vmax=max_val)
   plt.colorbar()
   plt.title("Phi, NCE")
   plt.xlabel("Dimension")
   plt.ylabel("Model")

   plt.subplot(1,2,2)
   plt.imshow(phi.reshape(2,3), vmin=min_val, vmax=max_val)
   plt.colorbar()
   plt.title("Phi, score matching")
   plt.xlabel("Dimension")
   plt.ylabel("Model")

   plt.suptitle("Phi, NCE vs SM with 1 model")
   logging.info("Saving the plot at: src/plots/plot_sm_vs_nce.png")
   plt.savefig('src/plots/plot_sm_vs_nce.png')
   plt.show()


if __name__ == "__main__":
   N = 1000 # samples
   nodes = 3
   K = 1 # number of models (or components)
   phi = [np.block([ 0, 0, 8*np.cos(np.pi), 8*np.sin(np.pi), 0, 0 ])]
   X, datamodel = sample_syndata_torusgraph(
         nodes = nodes,
         samples = N,
         phi = phi,
         fitFCM = False,
         fitPAD = True,
         fitPAS = False,
         return_datamodel = True
      )
   plot_nce_vs_sm(X, nodes, K, datamodel, N)
