import sys
sys.path.insert(0, '.')

from src.toolbox import NCE, TorusGraph, load_sample_data
import time
import os
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
import torch
import seaborn as sns

if __name__=="__main__":
    data, truevals = load_sample_data()
    data = torch.from_numpy(data).float()

    #plot distribution of true values
    fig, ax = plt.subplots()
    ax.hist(truevals, bins=100, color='blue', alpha=0.7)
    ax.set_title('Distribution of true values')
    plt.show()

    #boxplots
    # Using Z-score to identify outliers
    mean = torch.mean(data)
    std = torch.std(data)
    z_scores = (data - mean) / std
    outliers = torch.sum((z_scores < -3) | (z_scores > 3)).item()
    print(f"Number of outliers: {outliers}")# Typically, a Z-score greater than 3 is considered an outlier

    #print(f"Number of outliers: {torch.sum(outliers).item()}")



