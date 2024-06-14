## Script for creating synthetic data as described in [Klein2020]
## We need to make a script for creating synthetic data outside of Torus Graphs

## And another script for sampling in Torus Graphs.

## Much of this code is slighty modified and refactored from https://github.com/josue-orellana/pyTG/blob/main/pyTG.py
## All credit should go to them, yet they of course are not responsible for any mistakes

import numpy as np
from typing import List, Tuple, Any
from tqdm import tqdm

import sys
sys.path.insert(0, '.')

from src.data.utils import createNodePairs, drawVonMises, samplePhi, harmonicAddition, phiToParamGroups, phiParamGroupsToMats
from dataclasses import dataclass

@dataclass
class TorusGraphInformation:
    samples: int # trials
    nodes: int
    nodePairs: np.ndarray[np.ndarray[int, int]]
    nNodePairs: int
    params: int
    fitFCM: bool    # FirstCircularMoments
    fitPAD: bool    # PairwiseAngleDifferences
    fitPAS: bool    # PairwiseAngleSums


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
    return_datamodel:  bool = False
) -> np.ndarray: # of size (nodes, samlpes)

    """
    Samples from a torus graph.

    Parameters:
    nodes (int): The number of nodes in the graph.
    samples (int): The number of samples to generate.
    phi (np.ndarray): The initial phase angles.
    node_pairs (List[Tuple[int, int]]): The pairs of nodes in the graph.
    start_buffer (int): The number of initial samples to generate.
    sample_distance (int): The distance between samples.
    fit_fcm (bool): Whether to fit the first circular moments.
    fit_pad (bool): Whether to fit the pairwise angle differences.
    fit_pas (bool): Whether to fit the pairwise angle sums.
    return_datamodel (bool): Whether to return the data model.

    Returns:
    np.ndarray: The generated samples.
    Optional[TorusGraphInformation]: The data model.
    """

    if nodePairs is None: nodePairs = createNodePairs(nodes)
    # some way of setting basic phi if not set preiously
    if phi is None: phi = samplePhi(len(nodePairs), fitFCM, fitPAD, fitPAS)
    nNodePairs = int(nodes*(nodes - 1)/2)
    params = len(phi)
    totalSamples = startBuffer + samples * sampleDistance
    x = drawVonMises(0, 0, nodes)
    paramG = phiToParamGroups(phi, params, nodes, nNodePairs, fitFCM, fitPAD, fitPAS)
    matParamG = phiParamGroupsToMats(params, nodes, nodePairs, nNodePairs, paramG)
    cosMu, sinMu = matParamG["cosMu"], matParamG["sinMu"]
    mu = np.angle(cosMu +sinMu*1j)
    kap = np.abs(cosMu + sinMu*1j)
    indsNodesAll = np.arange(nodes)
    indsSampKeep = list(range(startBuffer,totalSamples,sampleDistance))
    iKeep = 0

    x1 = x.copy()
    x2 = x.copy()
    S1 = np.zeros((nodes, samples))
    for i in tqdm(range(totalSamples), "Sampling data..."):
        if i not in indsSampKeep:
            continue
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
            x1[k] = np.random.vonmises(resPhase, resAmp, 1)
        S1[:,iKeep]=x1
        iKeep += 1

    S2 = np.zeros((nodes, samples))
    iKeep = 0

    for i in tqdm(range(totalSamples), "Sampling data..."):
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
            x2[k] = np.random.vonmises(resPhase, resAmp, 1)
        if (i in indsSampKeep): 
            S2[:,iKeep]=x2
            iKeep += 1

    print(S1)
    print(S2)

    assert np.array_equal(S1, S2)
    if return_datamodel:
        datamodel = TorusGraphInformation(
                samples = samples,
                nodes = nodes,
                nodePairs = nodePairs,
                nNodePairs = nNodePairs,
                params = params,
                fitFCM = fitFCM,
                fitPAD = fitPAD,
                fitPAS = fitPAS
        )
        return S1, datamodel
    return S1

def sampleOutsideOfTorusGraph():
    drawVonMises()
    pass

if __name__ == "__main__":
    nodes = 4
    samples = 10
    output = sampleFromTorusGraph(
        nodes, 
        samples,
        fitFCM = False,
        fitPAD = True,
        fitPAS = False
    )
    print("### OUTPUT ###")
    print(f"You get a list of size ({nodes}, {samples}): {samples} samples for each of the {nodes} nodes")
    print("\n--- ###### ---\n")
    print(output)