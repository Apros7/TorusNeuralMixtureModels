
import sys
sys.path.insert(0, '.')

from src.data.synthetic_data import sampleFromTorusGraph
from src.parameterEstimation.scoreMatching import SM

def score_matching(phi, nodes, samples):
    X, datamodel = sampleFromTorusGraph(
        nodes = nodes,
        samples = samples,
        phi = phi,
        fitFCM = False,
        fitPAD = True,
        fitPAS = False,
        return_datamodel = True
    )
    newPhi, covPhi = SM(X, datamodel).compPhiHatAndCovPhiHat()
    return newPhi

score_matching()


