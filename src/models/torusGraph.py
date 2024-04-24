from src.data.synthetic_data import sampleFromTorusGraph

class TorusGraph():
    def __init__(self, data, estimation_func : Function = NCE):
        self.data = data
        self.estimate_params(estimation_func)

    def sample(samples : int = 100):
        return sampleFromTorusGraph(self.nodes, samples, phi = self.phi, nodePairs=self.nodePairs)