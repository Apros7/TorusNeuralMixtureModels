# TorusNeuralMixtureModels
Introduction to problem + solution

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

Estimate NCE loss over the amount of fitting torus graphs **MISSING**
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
Conclusion

## Citations
```
@article{Liu2023RingAW,
    title    = {Ring Attention with Blockwise Transformers for Near-Infinite Context},
    author   = {Hao Liu and Matei Zaharia and Pieter Abbeel},
    journal  = {ArXiv},
    year     = {2023},
    volume   = {abs/2310.01889},
    url      = {https://api.semanticscholar.org/CorpusID:263608461}
}
```

What to do:
- Lr nce mixture model
- Apply to real data
