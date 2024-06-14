import sys
sys.path.insert(0, '.')

import numpy as np
from pyTG import * 
import matplotlib.pyplot as plt
from src.data.synthetic_data import sampleFromTorusGraph
from src.data.utils import createNodePairs

def rayleigh(Y):
    n = Y.shape[1]
     
    # r is the magnitude of the complex mean
    r = np.abs(np.mean(np.exp(Y*1j), axis = 1))

    # compute Rayleigh's R
    # the magnitude of the complex sum
    R = n * r

    # compute Rayleigh's z 
    z = (R ** 2) / n

    # compute p value using approxation 
    pval = np.exp(np.sqrt(1 + 4 * n + 4 * (n ** 2 - R ** 2)) - (1 + 2 * n))

    return (pval, z)

def phaseLockingValue(X, nodePairs = None):
    if (nodePairs == None):
        if (X.shape[0]==2):
            nodePairs = np.array([[0,1]])
        else:
            nodePairs = createNodePairs(X.shape[0])
            #raise ValueError('When X has more than 2 rows, you need to include nodePairs. See :meth:`funsTG.compNodePairsNodes`.')

    Xi = X[nodePairs[:, 0], :]
    Xj = X[nodePairs[:, 1], :]
    Xdif = Xi - Xj

    plv = np.abs(np.mean(np.exp(1j*(Xdif)), axis =1))
    pVals, z = rayleigh(Xdif)

    return (plv, pVals) #hva er pVals???
    
X = sampleFromTorusGraph(3, 100, fitFCM = False, fitPAD = True, fitPAS = False)
print(phaseLockingValue(X))

# Compute PLV and p-values
plv, pVals = phaseLockingValue(X)

# Plotting
plt.figure(figsize=(10, 5))

# Plot PLV
plt.subplot(1, 2, 1)
plt.bar(range(len(plv)), plv, color='b')
plt.xlabel('Node Pairs')
plt.ylabel('Phase Locking Value (PLV)')
plt.title('Phase Locking Value (PLV) for Node Pairs')

# Plot p-values
plt.subplot(1, 2, 2)
plt.bar(range(len(pVals)), pVals, color='r')
plt.xlabel('Node Pairs')
plt.ylabel('p-value')
plt.title('p-values for Node Pairs')

plt.tight_layout()
plt.show()

plv_values, p_values = phaseLockingValue(X)

# Plotting
plt.figure(figsize=(10, 5))
plt.bar(range(len(plv_values)), plv_values, color='skyblue')
plt.xlabel('Node Pair Index')
plt.ylabel('Phase Locking Value (PLV)')
plt.title('Phase Locking Values between Node Pairs in Torus Graph')
plt.grid(True)
plt.show()