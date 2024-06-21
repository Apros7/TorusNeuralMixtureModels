import sys
sys.path.insert(0, '.')

from src.toolbox import NCE, TorusGraph, sample_syndata_torusgraph
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm


if __name__ == "__main__":
    N = 1000
    nodes = 3
    lr = 0.05
    nce_steps = 1000    

    Ks = [1,2,3,4,5]
    l_dict = {K: [] for K in Ks}

    for _ in tqdm(range(5)):
        data, datainformation = sample_syndata_torusgraph(
                nodes = nodes,
                samples = N,
                nModels = 3,
                return_datamodel = True,
            )
    
        for K in Ks:
            nce = NCE(
                nodes = nodes,
                K = K,
                steps = nce_steps,
                lr = lr,
            )
            TorusGraph(
                data = data,
                TGInformation = datainformation,
                nodes = nodes,
                samples = N,
                nModels = K,
                estimationMethod = nce,
            )
            l_dict[K].append(nce.losses[-1])


    losses_list = list(l_dict.values())
    losses_mean = [(np.mean(losses)) for losses in losses_list]
    plt.rcParams['font.family'] = 'Times New Roman'
    fig, ax = plt.subplots()
    plt.plot(np.array(Ks), losses_mean, marker = 'o')
    ax.set_xticklabels(list(l_dict.keys()))
    ax.set_xlabel('Number of models')
    ax.set_ylabel('Objective function')
    plt.title('Mean of objective function over sets of (1,2,...5) models')
    plt.show()

    #Skal det laves til abs værdier? - så får vi præcist det plot anders viste
    # losses_mean_abs = [abs(np.mean(losses)) for losses in losses_list]
    # plt.plot(Ks, losses_mean)
    # plt.title('Mean of objective function over sets of (1,2,...5) models')
    # plt.show()


    fig, ax = plt.subplots()
    ax.boxplot(losses_list, patch_artist=True)
    ax.set_xticklabels(list(l_dict.keys()))
    ax.set_xlabel('Number of models')
    ax.set_ylabel('Objective function')
    plt.title(f'Objective function over sets of (1,2,... {K}) models')
    plt.show()
    plt.savefig('src/plots/objective_function_k_models.png')



