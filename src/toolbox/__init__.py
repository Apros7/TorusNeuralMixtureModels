import sys
sys.path.insert(0, '.')

from src.parameterEstimation.scoreMatching import SM    # Score matching
from src.parameterEstimation.NCE import NCE             # Noise contrastive estimation
from src.toolbox.data import sample_syndata_torusgraph, syndata_true_labels, estimate_uniform_noise, load_sample_data
from src.data.synthetic_data import TorusGraphInformation
from src.toolbox.eval import calc_NMI, visualize_phase_coherens
from src.toolbox.torus import TorusGraph