from typing import List, Tuple
import numpy as np
import torch
import logging

from src.data.synthetic_data import sampleFromTorusGraph, TorusGraphInformation
from src.data.real_data import load_sample_data

NMODELS_TO_PHI = {
    1: [
        np.block([ 0, 0, 8*np.cos(np.pi), 8*np.sin(np.pi), 0, 0 ])
    ],
    2: [
        np.block([ 0, 0, 8*np.cos(np.pi), 8*np.sin(np.pi), 0, 0 ]), 
        np.block([ 0, 0, 8*np.sin(np.pi), 8*np.cos(np.pi), 0, 0 ])
    ],
    3: [
        np.block([ 0, 0, 8*np.cos(np.pi), 8*np.sin(np.pi), 0, 0 ]), 
        np.block([ 0, 0, 8*np.sin(np.pi), 8*np.cos(np.pi), 0, 0 ]), 
        np.block([ 0, 0, -8*np.cos(np.pi), -8*np.sin(np.pi), 0, 0 ])
        #np.block([ 0, 8*np.cos(np.pi), 0, 0, 8*np.sin(np.pi), 0 ])
    ],
}

def sample_syndata_torusgraph(
    nodes   : int, 
    samples : int, 
    nModels: int = 1,
    phi     : List[np.ndarray] = None,
    nodePairs: List[Tuple[int, int]] = None,
    startBuffer: int = 500,
    sampleDistance: int = 100,
    fitFCM: bool = False,    # FirstCircularMoments
    fitPAD: bool = True,     # PairwiseAngleDifferences
    fitPAS: bool = False,    # PairwiseAngleSums
    return_datamodel:  bool = False
) -> Tuple[List[Tuple[np.ndarray, ]], np.ndarray]:
    """
    
    """
    if phi is None and nModels > 3: raise NotImplementedError("Default phi not implemented for higher than 3 models. You can always specify your own to Phi array to prevent this error.")
    if phi is None: 
        phi = NMODELS_TO_PHI[nModels]
        logging.warning(f"You did not set phi, so setting it to: {phi}")
    if len(phi) != nModels: raise ValueError(f"The number of models to drawn from (nModels) has to match the length of the phi array (should be {nModels}, but is {len(phi)}")
    all_samples = []
    for i in range(nModels):
        all_samples.append(sampleFromTorusGraph(
            nodes = nodes,
            samples = samples,
            phi = phi[i],
            nodePairs = nodePairs,
            startBuffer = startBuffer, 
            sampleDistance = sampleDistance,
            fitFCM = fitFCM,
            fitPAD = fitPAD,
            fitPAS = fitPAS,
            return_datamodel = return_datamodel
        ))
    logging.warning("You might get a RuntimeError in the NCE objective function estimation with 'log_J2_denom'\
        If that happens, modify your data after sampling but before giving it to the torus graph with:\
            X = torch.from_numpy(X).float().T")
    if nModels == 1: return all_samples[0]
    samples_drawn = np.concatenate([x[0].T for x in all_samples]).T
    try:
        all_phi = np.array([x[1].phi for x in all_samples])
    except AttributeError: 
        raise AttributeError("Trying to create all phi, but did not work. You most likely have to set 'return_datamodel = True' when initializing the TorusGraph class")
    full_TGInformation = TorusGraphInformation(
        samples = samples * nModels,
        phi = all_phi,
        fitFCM = fitFCM,
        fitPAD = fitPAD,
        fitPAS = fitPAS,
    )
    return (samples_drawn, full_TGInformation)

def syndata_true_labels(nModels: int, samples: int):
    """
    When samples from nModels different torusgraphs, the underlying groups are already known
    """
    return np.repeat(np.arange(nModels), samples)

def estimate_uniform_noise(nodes: int, samples: int):
    return torch.rand(samples, nodes) * 2 * torch.tensor(np.pi)