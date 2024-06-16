
import sys
sys.path.insert(0, '.')

from src.toolbox import TorusGraph, NCE
import matplotlib.pyplot as plt

if __name__ == "__main__":
    N = 1000 # samples
    nodes = 3
    K = 1 # single model
    nce_steps = 2000
    lr = 0.1

    nce = NCE(
        nodes = nodes,
        K = K,
        steps = nce_steps,
        lr = lr
    )
    TorusGraph(
        nodes = nodes,
        samples = N,
        nModels = K,
        estimationMethod = nce,
    )
    plt.plot(nce.losses)
    plt.title(f"NCE loss over {nce_steps} steps with lr = {lr} for a single torus graph")
    plt.show()

