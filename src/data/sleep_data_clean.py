import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import glob
import sys

sys.path.insert(0,'.')


# første element i phases indeholderne målingerne for første element i stages.

sub1 = pd.read_csv('src/data/phases_7/sub-01_mainlysleep_phases.txt', header=None)
sub1.columns = ['mainlysleep_phases']
sub2 = pd.read_csv('src/data/phases_7/sub-04_mainlysleep_phases.txt', header=None)
