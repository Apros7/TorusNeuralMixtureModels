## Script for creating synthetic data as described in [Klein2020]
## We need to make a script for creating synthetic data outside of Torus Graphs

## And another script for sampling in Torus Graphs.

## Much of this code is slighty modified and refactored from https://github.com/josue-orellana/pyTG/blob/main/pyTG.py

import numpy as np

from typing import List, Tuple, Any

def createNodePairs(nodes: int) -> np.ndarray[np.ndarray[int, int]]:
    numNodePairs = int(nodes*(nodes-1)/2)
    nodePairsNodes = np.zeros((numNodePairs, 2), dtype=int)
    inc = 0
    for i in range(nodes):
        for j in range(i+1,nodes):
            nodePairsNodes[inc,:] = [i, j]
            inc +=1
    return nodePairsNodes

def drawVonMises(mu : float, kappa : float = 0.0, samples : int = 1) -> np.ndarray:
    return np.random.vonmises(mu, kappa, size = samples)

def samplePhi(
    nNodePairs: int, 
    fitFirstCircularMoments: bool, 
    fitPairwiseAngleDifferences: bool, 
    fitPairwiseAngleSums: bool
) -> np.ndarray:
    # size should be:
    # 2 * number of nodePairs for each fitting method
    return np.array([[0, 0, 0, 0], [8*np.cos(np.pi), 8*np.sin(np.pi), 0, 0]])

def sampleFromTorusGraph(
    nodes   : int, 
    samples : int, 
    phi     : np.ndarray = None,
    nodePairs: List[Tuple[int, int]] = None,
    startBuffer: int = 500,
    sampleDistance: int = 100,
    fitFCM: bool = True,    # FirstCircularMoments
    fitPAD: bool = True,    # PairwiseAngleDifferences
    fitPAS: bool = True,    # PairwiseAngleSums
):
    if nodePairs is None: nodePairs = createNodePairs(nodes)
    # some way of setting basic phi if not set preiously
    if phi is None: phi = samplePhi(len(nodePairs), fitFCM, fitPAD, fitPAS)
    nNodePairs = int(nodes*(nodes - 1)/2)
    params = len(phi)
    totalSamples = startBuffer + samples * sampleDistance
    x = drawVonMises(0, 0, nodes)
    paramG = phiToParamGroups(phi, numSamp, selMode)
    matParamG = phiParamGroupsToMats(numSamp, paramG, nodePairs)
    mu = np.angle(cosMu +sinMu*1j)
    kap = np.abs(cosMu + sinMu*1j)
    indsNodesAll = np.arange(nodes)
    indsSampKeep = list(range(startBuffer,totalSamples,sampleDistance))
    iKeep = 0

    S = np.zeros((nodes, samples))
    for i in range(totalSamples):
        for k in range(nodes):
            smDelta = np.concatenate(( x[:k]     - np.pi/2, 
                                       x[(k+1):] + np.pi/2 ))

            indsNoK = np.concatenate(( indsNodesAll[:k], indsNodesAll[(k+1):] ))

            amps = np.concatenate(( kap[k:(k+1)], 
                                    matParamG['cosMuDif'][k,indsNoK],
                                    matParamG['sinMuDif'][k,indsNoK],
                                    matParamG['cosMuSum'][k,indsNoK],
                                    matParamG['cosMuSum'][k,indsNoK] )) 
            phases = np.concatenate(( mu[k:(k+1)],
                                    x[indsNoK],
                                    smDelta,
                                    -x[indsNoK],
                                    -x[indsNoK] + np.pi/2 ))                                                    
            resAmp, resPhase = harmonicAddition(amps, phases)
            x[k] = np.random.vonmises(resPhase, resAmp, 1)
        if (i in indsSampKeep): 
            S[:,iKeep]=x
            iKeep += 1
    return S

def sampleOutsideOfTorusGraph():
    drawVonMises()
    pass

if __name__ == "__main__":
    nodes = 4
    samples = 1000
    sampleFromTorusGraph(nodes, samples)