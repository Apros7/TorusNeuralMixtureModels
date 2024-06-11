from src.data.synthetic_data import sampleFromTorusGraph
from src.parameterEstimation.NCE import NCE
from src.parameterEstimation.scoreMatching import SM

class TorusGraph():
    def __init__(self, data, estimation_func: NCE or SM = NCE): # estimation_func can be NCE or SM
        self.data = data
        self.phi, self.buffer = self.estimate_params(estimation_func)
        self.nodes = 0
        self.nodePairs = 0
        
    def estimate_phi(self):
        self.phi, self.c = "hey", "hey"

    def sample(self, samples : int = 100):
        return sampleFromTorusGraph(self.nodes, samples, phi = self.phi, nodePairs=self.nodePairs)