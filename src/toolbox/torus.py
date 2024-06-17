import sys
sys.path.insert(0, '.')

from src.parameterEstimation.scoreMatching import SM    # Score matching
from src.parameterEstimation.NCE import NCE             # Noise contrastive estimation
from src.toolbox.data import sample_syndata_torusgraph, syndata_true_labels, estimate_uniform_noise, load_sample_data
from src.data.synthetic_data import TorusGraphInformation
from src.toolbox.eval import calc_NMI, visualize_phase_coherens, classify_points, ohe, adjust_pred_labels

import numpy as np
import logging
import matplotlib.pyplot as plt
import torch
from sklearn.metrics import accuracy_score
from collections import Counter

class TorusGraph():
    """
    Required:
    - nodes (int): number of nodes in your data
    - samples (int): number of samples in your data
    
    Optional: 
    - nModels (int, default: 1): Number of models. If k = 1, then the parameters estimation can be either NCE or SM, else it is NCE.
    - estimationMethod (NCE or SM, default: SM): If k > 1, then NCE has to be used
    - data (np.ndarray, default: None): If none then samples from synthetic data using k torus graphs
    - TGInformation (TorusGraphInformation, default: None): If SM is used as estimation method, then this class has to be filled out

    Params:
    - .info: initialization information from the torus graph
    - .data: data used by the model
    - .estimationMethod: the method used for estimation
    - .true_vals: the true labels for evaluation
    - .TGInformation: more information available for how the sampling was made

    Methods:

    When initialized, the class will sample data if not already avaliable and run its specified estimation method.
    Afterwards the results can be evaluated by running 'evaluate'. This will:
    1. Show a plot of the phi phase-coherens of the different nodes
    2. Display the NMI value compared to real data.

    The loss function for the NCE algorithm can now also be called by doing NCE.plot_loss().

    Progress bars:
    
    Both the sampling and NCE will show progress bars to give information of progress. To disable them set env var to True:
    ```
    os.environ['DISABLE_TQDM'] = 'True'
    ```
    """
    def __init__(
        self,
        nodes: int,
        samples: int = 100000,
        nModels: int = 1,
        estimationMethod: NCE or SM = None,
        data: np.ndarray = None,
        noise: np.ndarray = None,
        TGInformation: TorusGraphInformation = None,
        true_vals: np.ndarray = None,
        return_datamodel: bool = False,
    ):
        self.true_vals, self.data, self.TGInformation, self.estimationMethod, self.noise = true_vals, data, TGInformation, estimationMethod, noise
        if data is None:
            logging.info("No data was set, so sampling synthetic data with default settings. If you wish to change those, sample the data beforehand. ")
            if nModels == 1 or return_datamodel:
                self.data, self.TGInformation = sample_syndata_torusgraph(nodes = nodes, samples = samples, nModels = nModels, return_datamodel = True)
            else:
                self.data = sample_syndata_torusgraph(nodes = nodes, samples = samples, nModels = nModels, return_datamodel = False)
            self.true_vals = syndata_true_labels(nModels = nModels, samples = samples)
        if self.TGInformation and self.TGInformation.samples:
            samples = self.TGInformation.samples
        if self.true_vals is None:
            logging.info("You did not set true vals, so estimating them from parameters, this could be dangerous")
            self.true_vals = syndata_true_labels(nModels = nModels, samples = samples // nModels)
        if estimationMethod is None and self.TGInformation is not None:
            logging.info("You did not set an estimation method, so it defaults to Score Matching")
            self.estimationMethod = SM(self.data, self.TGInformation)
        if isinstance(self.estimationMethod, SM) and self.TGInformation is None:
            logging.warning("You wanted to estimate using SM, but no TorusGraphInformation was set, so switching to using NCE for parameter estimation.")
            self.estimationMethod = NCE(nodes, nModels)
        if isinstance(self.estimationMethod, NCE):
            if self.noise is None: self.noise = estimate_uniform_noise(nodes, samples)
            self.estimationMethod.run(self.data, self.noise)
        elif isinstance(self.estimationMethod, SM):
            self.estimationMethod.run()
        else:
            raise NotImplementedError("Not implemented for estimation method", type)
        self.info = {"nodes": nodes, "samples": samples, "nModels": nModels, "estimation method": self.estimationMethod.__class__.__name__, "returned datamodel": return_datamodel}
    
    def evaluate(self):
        adjusted_pred_labels = self.get_preds()
        print("The distribution is:\n---------------")
        print("Preds: ", Counter(adjusted_pred_labels))
        print("Trues: ", Counter(self.true_vals))
        return accuracy_score(adjusted_pred_labels, self.true_vals)
        # return calc_NMI(pred_labels_ohe, self.true_vals)#, accuracy_score(pred_labels, self.true_vals)

    def visualize(self):
        adjusted_pred_labels = self.get_preds()
        distributions = {}
        for k in range(self.info["nModels"]):
            preds_for_k_idx = np.where(adjusted_pred_labels == k)
            preds_distribution = Counter(self.true_vals[preds_for_k_idx])
            print(preds_distribution)
            distributions[k] = [preds_distribution.get(i, 0) for i in range(self.info["nModels"])]

        print(distributions)
        fig, ax = plt.subplots()
        x_values = list(distributions.keys())
        bar_width = 0.8 / self.info["nModels"]
        for i in range(self.info["nModels"]):
            bars = [distributions[k][i] for k in x_values]
            ax.bar([x + i * bar_width for x in range(len(x_values))], bars, bar_width, label=f"Model {i}")

        # Set x-axis labels
        ax.set_xticks([x + bar_width * (self.info["nModels"] - 1) / 2 for x in range(len(x_values))])
        ax.set_xticklabels(x_values)
        ax.set_title("Distribution of Predictions by Model")
        ax.set_xlabel("Model Index")
        ax.set_ylabel("Count")
        ax.legend()
        plt.show()

    def get_preds(self):
        if not self.estimationMethod.return_log_prop_data:
            raise NotImplementedError("You need to set return log prop data to True in your estimation method")
        if self.true_vals is None:
            raise NotImplementedError("You need to set true vals before being able to run eval")
        pred_labels = classify_points(self.estimationMethod, self.estimationMethod.log_prop_data, self.info["samples"]).detach().numpy()
        print(pred_labels.shape, self.true_vals.shape)
        adjusted_pred_labels = adjust_pred_labels(pred_labels, self.info["nModels"], self.true_vals)
        return adjusted_pred_labels



