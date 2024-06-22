# TorusNeuralMixtureModels
The objective of this paper is to investigate if the existing torus graph model for capturing conditional associations amongst oscillating signals, for example from brain measurements can be extended to a torus graph mixture model to capture the same associations over time using noise contrastive estimation for estimating its parameters (Klein 2020, Torus graphs for multivariate phase coupling analysis). The parameters for a singular torus graph are estimated using score matching, whereas the parameters for the torus graph mixture models are estimated using noise contrastive estimation. To enable comparison between the methods, the score of normalized mutual information is used. This paper finds that using score matching for a singular torus graph captures the conditional associations better than using noise contrastive estimation for a singular torus graph. Extending it to a mixture model, it is found that the torus graph mixture models capture these conditional association using noise contrastive estimation on synthetic data, however applying the same methods on EEG-measurements from different brain areas yielded poorer results and the model did not capture any noteworthy associations from the different brain areas. Conclusively the torus graph mixture model did not capture conditional association from brain areas accurately using noise contrastive estimation on real data. The main take away is that it is possible to extend the torus graph to a torus graph mixture model to capture conditional associations in oscillating signals over time, however noise contrastive estimation might not be the parameter estimation method best suited for the task without further improvement of the method.

## Set-up
We chose to use [make](https://www.gnu.org/software/make/) to automate our processes and improve ease-of-use when using our repo. See [here](https://stackoverflow.com/questions/32127524/how-to-install-and-use-make-in-windows) to install on windows or use ```brew install make``` if you're on a macbook.

First, run these commands to setup the project:
```
python3 -m venv .venv
source .venv/bin/activate
make
```

You are now ready to reproduce our results or work with our toolbox.

## Reproduce results

In all cases, the parameters can be adjusted in the file refered in the makefile if necessary.
The exact same code will be executed as done in our report, yet the data and initialization of the NCE is randomized, hence slightly different results will show.

**On syn_data:**

Find the best learning rate for the NCE algorithm using a single Torus Graph
```
make best_lr_nce_tg:
```

Find the best learning rate for the NCE algorithm using a Torus Graph Mixture Model
```
make best_lr_nce_tgmm:
```

Visualize phi for a single Torus Graph using SM vs TG NCE
```
make sm_vs_tg_viz
```

Quantify the differens in phi for a single Torus Graph using SM vs TG NCE compared to the sampled phi
```
make sm_vs_tg_boxplot
```

See the NCE loss over time when estimating parameters 
```
make nce_loss
```

See the NCE loss over the amount of fitting torus graphs
```
make nce_loss_tgmm
```

See the distribution of classes on TGMM components
```
make see_preds_data
```

**on real data**

Find the best learning rate for the NCE algorithm using a Torus Graph Mixture Model on real data 
```
make real_best_lr_nce_tgmm:
```

Quantify the differens in output class between true output and TGMM
```
make real_tgmm_boxplot
```

Visualize phi for a single Torus Graph using TGMM NCE 
```
make real_tgmm_viz
```

Estimate NCE loss over the amount of fitting torus graphs
```
make real_nce_loss_tgmm
```

See the distribution of classes on TGMM components
```
make real_see_preds_data
```

## Toolbox

The toolbox tries to generalize our approach into functions to be used for other purposes.
In some cases in simply documents another function called in our repo, as this has all the functionality needed to perform the action.
In other cases the function exists somewhere else, but is imported into the toolbox to allow one to follow the directory to the original function:

The following are available:

## Conclusions

Throughout our projekt the following research questions are aimed to be answered:

- How can torus graphs be extended to torus graph mixture models?
- How good are torus graph mixture models (TGMMs) at estimating oscillating phase coherence in signals?
- How can these methods (TGMMs) be applied to find new conditional associations in data?

**Conclusion:**
Throughout the project the above mentioned research questions has been answered. Firstly an extension of the existing torus graph model to a torus graph mixture model was made and then employed with noise contrastive estimation for torus graphs mixture models to find its parameters. Extending the existing torus graph mixture model was done by following the mathematics exerted in section 2.3 with inspiration from Matsuda et al. 2019 \cite{Matsuda2019} and Klein et al. 2020 \cite{Klein2020}. When performing statistical tests, it was found that there was a significant difference in the estimated parameters for the torus graph sampled with synthetic data, concluding that NCE was a poor parameter estimation for a singular torus graph. This could be explained by the choice of noise distribution.

Investigating the performance of torus graph mixture models (TGMM) with noise contrastive estimation (NCE) yielded different results. Firstly with synthetic data, it was found that NCE for TGMM's performed relatively well and managed to find conditional association amongst the oscillating signals, thus enabling modelling over time. However, when extendeding it to EEG-measurements from brain areas, the mixture model with NCE did not manage to find noteworthy conditional associations. It found an association between the ventral attention area and the limbic area, but this is deemed inapplicable due to the general poor performance of the model using NCE. Firstly this could be explained by the choice of noise distribution not being adequately related to the actual brain data. it could also be the vague distinguish in classes in the actual brain data. Thirdly, one could conclude that using NCE for a torus graph mixture model is not applicable on real data, hence another method of parameter estimation might be better. In order to firmly conclude on this matter, further investigation into noise distribution and multiple data sets would be insightful. Conclusively torus graph mixture models with noise contrastive parameter estimation does not meet the expectations of this paper, and is thus not applicable on the data set \cite{dataset}. However, since it proved well on the synthetic data, one cannot discard the method all together. Therefore further investigation into different methods for parameter estimation for mixture models would be beneficial, as well as noise distributions and data sets as earlier mentioned. It is important to note that this technological advancement comes with both positive advantages but also various ethical dilemmas to consider as mentioned in the discussion.

The main objective was to find out if the torus graph mixture model could capture the conditional associations from oscillating signals over time efficiently and accurately using noise contrastive estimation. After analyzing the results, it can be concluded that the torus graph mixture model did not capture conditional association from brain areas accurately using noise contrastive estimation, hence opening the field for other parameter estimation methods for mixture models. The main take away is that it is possible to extend the torus graph to a torus graph mixture model to capture conditional associations in oscillating signals over time, however noise contrastive estimation might not be the parameter estimation method best suited for the task.

## Citations
```
@online{Klein2020,
    author    = "Natalie Klein et al.",
    year      = "2020",
    title     = "Torus Graphs for Multivariate Phase Coupling Analysis",
    url       = "https://www.ncbi.nlm.nih.gov/pmc/articles/PMC9812283/",
    keywords  = "torus,graphs"
}

@online{Matsuda2019,
    author    = "Takeru Matsuda et al.",
    year      = "2019",
    title     = "Estimation of NonNormalized Mixture Models",
    url       = "https://proceedings.mlr.press/v89/matsuda19a/matsuda19a.pdf"
}

@online{TorusGraphCode,
  title = {Code for Torus Graphs},
  publisher = {Github, Natalie Klein},
  year = {2021},
  author = {Wikipedia contributors},
  url = {https://github.com/josue-orellana/pyTG/}
}

@online{dataset,
  title = {Simultaneous EEG and fMRI signals during sleep from humans},
  publisher = {OpenNeuro},
  year = {2023},
  author = {Yameng Gu, Feng Han, Lucas E. Sainburg, Margeaux M. Schade, Xiao Liu},
  url = {https://openneuro.org/datasets/ds003768/versions/1.0.11}
}

@online{NMI,
  title = {Normalized mutual information},
  publisher = {MDPI},
  year = {2017},
  author = {Tarald O. Kvålseth},
  url = {https://www.mdpi.com/1099-4300/19/11/631}
}
```