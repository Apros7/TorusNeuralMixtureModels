
import numba as nb
import numpy as np

@nb.njit(parallel=True)
def harmonicAddition(amps, phases):
    # Assuming harmonicAddition is a simple function that can be parallelized
    resAmp = np.zeros(amps.shape[1])
    resPhase = np.zeros(amps.shape[1])
    for i in nb.prange(amps.shape[1]):
        resAmp[i], resPhase[i] = harmonicAddition_serial(amps[:, i], phases[:, i])
    return resAmp, resPhase

@nb.njit
def harmonicAddition_serial(amps, phases):
    # Implement the harmonicAddition function here
    # This is a placeholder, you need to implement the actual logic
    return amps.sum(), phases.sum()

@nb.njit(parallel=True)
def vectorized_loop(kap, x, mu, matParamG, indsNodesAll, nodes):
    x_new = np.zeros_like(x)
    for i in nb.prange(nodes):
        smDelta = np.concatenate((x[:i] - np.pi/2, x[(i+1):] + np.pi/2))
        indsNoK = np.concatenate((indsNodesAll[:i], indsNodesAll[(i+1):]))
        amps = np.concatenate((kap[i:(i+1)], 
                              matParamG['cosMuDif'][i,indsNoK],
                              matParamG['sinMuDif'][i,indsNoK],
                              matParamG['cosMuSum'][i,indsNoK],
                              matParamG['cosMuSum'][i,indsNoK]))
        phases = np.concatenate((mu[i:(i+1)],
                                 x[indsNoK],
                                 smDelta,
                                 -x[indsNoK],
                                 -x[indsNoK] + np.pi/2))
        resAmp, resPhase = harmonicAddition(np.array([amps]), np.array([phases]))
        x_new[i] = np.random.vonmises(resPhase, resAmp, 1)
    return x_new