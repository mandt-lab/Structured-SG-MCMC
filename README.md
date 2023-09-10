# Structured Stochastic Gradient MCMC (S-SGMCMC)
Repository for implementation of the method proposed in [Structured Stochastic Gradient MCMC](https://arxiv.org/abs/2107.09028) by Antonios Alexos, Alex Boyd, and Stephan Mandt published in ICML 2022. S-SGMCMC is a MCMC technique that estimates an empirical posterior distribution using randomly sampled mini-batches of data. Time to convergence is improved with this method over other SGMCMC techniques due to imposing independence assumptions across different subsets of parameters, resulting in a simpler (variational) posterior to approximate.

### Instructions
Included in this repository is the `lvi` package. Within, the main file to interface with is `lvi/components.py`. The proposed method, along with other traditional SGMCMC techniques, are implemented and tied to model objects that handle sampling, prediction, and training. The main (abstract) parent class is the `StochasticNetwork`. Any subclass model that inherits from this (e.g., currently implemented are a multi-layer feedforward network `FF_BNN`, ResNet `BayesianResNet`, and more vanilla CNN `SVHN_BCNN`) will require the following information:
- Parameter grouping: Every parameter in a network will have an assigned integer ID that indicates what group it belongs to. Should two parameters have the same ID, then they are allowed to have their correlation modeled via the posterior. Typically, the subclasses have various options for determining this, such as for `FF_BNN` we can choose to either group the parameters by layer (`group_by_layers=True`) or we can set a maximum number of groups `max_groups=N` and then randomly assign an ID of 1,...,N based off a specific protocol (either `use_random_groups`, `use_permuted_groups`, or `use_neuron_groups`).
- `chain_length`: The proposed method requires keeping previous samples in memory to routinely sample from. As such, the amount that we are able to hold at once is determined by the `chain_length` attribute. Should we try to sample more than these, the model will overwrite the oldest samples with the new ones. It should be noted that imposing the independence assumption on the posterior requires a sufficiently long chain.
- `thinning_rate`: This determines how many iterations/gradient updates should be executed before saving the model's current state as a sample in the chain.
- `dropout_distribution`: Also proposed in the paper is a further approximation that enables better scaling to higher number of total parameter groups. This is enabled by specifying a `dropout_distribution` to either be `"uniform"`, `"beta"` with concentration parameters `beta_a` and `beta_b`, or `"bernoulli"` with probability parameter `dropout_prob`. Note that should this be enabled, it is recommended to use a `vi_batch_size>1` during training, as this controls the amount of Monte-Carlo estimates used for a given update.

The rest of the arguments expected are architecture specific. Once a model has been instantiated, it must first call two methods prior to training:
- `model.initalize_optimizer(...)`: The model itself is responsible for the optimizer and this method initializes and setups the optimizer to keep track of the appropriate parameters for updating. The important arguments are:
    - `update_determ` and `update_stoch`: A given model may have a mix of parameters that are treated as deterministic points and as stochastic random variables. Setting these arguments to `True` enables optimization of these parameters respectively.
    - `lr`: Learning rate.
    - `sgd`, `rmsprop`: Two stochastic gradient (not MCMC) optimization methods. Setting one to `True` will use the corresponding algorithm for optimization.
    - `sgld`, `psgld`, and `sghmc`: Three SGMCMC sampling methods. Setting one to `True` will use the corresponding algorithm for sampling, where a given sample is generated via the model's `training_step` method call.
- `model.init_chains()`: This is done to give a default value to the chains to sample from once training starts.

Once both of these methods are called, then sampling can procede by simply executing `model.training_step(...)` with the following arguments:
- `batch`: A tuple containing the input feature tensor first and the target tensor second.
- `N`: Total number of data points; this is used to properly scale the loss.
- `vi_batch_size`: This determines the number of Monte Carlo samples used in conjunction with dropout. This should be `None` if dropout is not being used.
- `deterministic_weights`: Setting to `True` disables adding noise and effectively mimics using stochastic gradient optimization for finding the MAP. This is recommended for the beginning portion of training to help with stability.
Samples generated during training are automatically pushed onto the model's chains of prior samples. These can be accessed by calling `model.get_chains()`.

Example usages of this package can be seen in the notebooks `do_lvi_tests_lr.ipynb` and `do_lvi_test_toy.ipynb`, as well as the main execution script for running experiments under `main.py`. 

### General Repo Structure

The main files of the package are structed as follows:
- `lvi/`
    - `components.py`: Implementations of stochastic versions of tensors and models that allow for sampling and tracking these samples in fixed length chains.
    - `optimizers.py`: Implementations of SGMCMC methods used in conjunction with our proposed approach.
    - `utils.py`: Helper functions and utilities for logging, training, loading data, etc.

### For Bibtex Citations
If you find our work helpful in your own research, please cite our paper with the following:
```
@InProceedings{alexos22sgmcmc,
  title={Structured Stochastic Gradient {MCMC}},
  author={Alexos, Antonios and Boyd, Alex J and Mandt, Stephan},
  booktitle={Proceedings of the 39th International Conference on Machine Learning},
  pages={414--434},
  year={2022},
}
```
