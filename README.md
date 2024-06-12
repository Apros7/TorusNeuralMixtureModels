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

Then, run the tests to make sure the software works as expected:
```
make test
```

## Reproduce results
You can reproduce a single run of our NCE & SM parameter estimation results on a Torus Graph from synthetic data by running
```
make tg_est_single
```
or use cross validation
```
make tg_est_cv
```

You can also see our results of a mixture model from synthetic data:
```
make mixture_syn_data
```

or from real data:
```
make mixture_real_data
```


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

Need 
- [ ] Fix NCE
- [ ] Visualize NCE results
- [ ] SM
- [ ] Visualize SM results
- [ ] Evaluation
- [ ] 

Marginal uniform første led gå rud