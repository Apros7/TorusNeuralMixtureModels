from src.data.synthetic_data import sampleFromTorusGraph

class TorusGraph():
    def __init__(self, data, estimation_func): # estimation_func can be NCE or SM
        self.data = data
        self.estimate_params(estimation_func)
        self.nodes = 0
        self.nodePairs = 0
        self.estimate_phi()
        
    def estimate_phi(self):
        self.phi, self.c = "hey", "hey"

    def sample(self, samples : int = 100):
        return sampleFromTorusGraph(self.nodes, samples, phi = self.phi, nodePairs=self.nodePairs)