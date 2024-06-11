import numpy as np
from typing import List, Tuple, Any

## Much of this code is slighty modified and refactored from https://github.com/josue-orellana/pyTG/blob/main/pyTG.py
## All credit should go to them, yet they of course are not responsible for any mistakes

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
    kap = np.random.randint(0, 10, nNodePairs)
    phi = np.block([kap, kap*0])
    return phi

def harmonicAddition(amps, phases) -> tuple:
    # inputs are numpy arrays
    bx = sum(amps*np.cos(phases))
    by = sum(amps*np.sin(phases))

    resAmp = np.sqrt(bx**2 + by**2)
    resPhase = np.arctan2(by, bx)

    return (resAmp, resPhase)

def phiToParamGroups(phi, phi_params, nodes, nodepairs, fitFCM, fitPAD, fitPAS):

    phi = np.copy(phi) # to avoid changing input phi
    numParam = 0
    params = dict()
    if fitFCM:
        params['cosMu'], phi  = phi[:nodes], phi[nodes:]
        params['sinMu'], phi  = phi[:nodes], phi[nodes:] 
        numParam += 2*nodes
    else:
        params['cosMu'] = np.zeros(nodes)
        params['sinMu'] = np.zeros(nodes)

    if fitPAD:
        params['cosMuDif'], phi = phi[:nodepairs], phi[nodepairs:]
        params['sinMuDif'], phi = phi[:nodepairs], phi[nodepairs:]
        numParam += 2*nodepairs
    else:
        params['cosMuDif'] = np.zeros(nodepairs)
        params['sinMuDif'] = np.zeros(nodepairs)

    if fitPAS:
        params['cosMuSum'], phi = phi[:nodepairs], phi[nodepairs:]
        params['sinMuSum'], phi = phi[:nodepairs], phi[nodepairs:]
        numParam += 2*nodepairs
    else:
        params['cosMuSum'] = np.zeros(nodepairs)
        params['sinMuSum'] = np.zeros(nodepairs)
    
    if phi_params != numParam :
        raise ValueError(
        "The number of parameters in phi doesn't agree with the provided selMode, which expects " \
        + str(numParam) + f" parameters, but is {phi_params}.")
        
    return params

def phiParamGroupsToMats(params, nodes, nodePairs, nNodePairs, paramG):
    # Only the pairwise components become matrices
    matParamG = {'cosMu': paramG['cosMu'],
                 'sinMu': paramG['sinMu'],
                 'cosMuDif': np.zeros((nodes, nodes)),
                 'sinMuDif': np.zeros((nodes, nodes)),
                 'cosMuSum': np.zeros((nodes, nodes)),
                 'sinMuSum': np.zeros((nodes, nodes))}

    # Step through all node-pairs
    for r in range(nNodePairs):
        rPair = (nodePairs[r,0], nodePairs[r,1])
        matParamG['cosMuDif'][rPair] = paramG['cosMuDif'][r]
        matParamG['sinMuDif'][rPair] = paramG['sinMuDif'][r]
        matParamG['cosMuSum'][rPair] = paramG['cosMuSum'][r]
        matParamG['sinMuSum'][rPair] = paramG['sinMuSum'][r]

    # make matrices symmetric
    matParamG['cosMuDif'] += matParamG['cosMuDif'].T
    matParamG['sinMuDif'] += matParamG['sinMuDif'].T
    matParamG['cosMuSum'] += matParamG['cosMuSum'].T
    matParamG['sinMuSum'] += matParamG['sinMuSum'].T

    return matParamG