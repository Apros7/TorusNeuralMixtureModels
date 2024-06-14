import numpy as np
import pyTG
import matplotlib.pyplot as plt

X = np.random.rand(3, 100)*2*np.pi

out = pyTG.torusGraphs(X, selMode=(False, True, False)) #en samling af variable, som er NodeGraphs, kig på 5 som er phihat
phi_hat = out[-2]

# Given arrays
Phi_cos = np.zeros((3,3))
Phi_sin = np.zeros((3,3))
print(phi_hat)

# Given values to assign
phi_hat = [phi_hat[0], phi_hat[1], phi_hat[2], phi_hat[3], phi_hat[4], phi_hat[5]] 

# List of assignments: (i, j, phi_hat_index, matrix)
assignments = [
    (0, 1, 0, Phi_cos),
    (0, 2, 1, Phi_sin),
    (1, 2, 2, Phi_cos),
    (0, 1, 3, Phi_sin),
    (0, 2, 4, Phi_cos),
    (1, 2, 5, Phi_sin)
]

# For loop to perform the assignments
for i, j, index, matrix in assignments:
    matrix[i, j] = phi_hat[index]
print(matrix)

plt.figure()
fig,axs = plt.subplots(1,2)
pos = axs[0].imshow(Phi_cos)
fig.colorbar(pos, ax=axs[0])
pos = axs[1].imshow(Phi_sin)
fig.colorbar(pos, ax=axs[1])
plt.savefig('plot_sm.png')

a=7

#matricer symetrisk - kun værdier i den øvere del eller nedre del transporneret
#farve hvor stærk fasekoblingen er, viser de forskellige dele af hjerne området