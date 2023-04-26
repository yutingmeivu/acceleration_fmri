# Acceleration of quantitative modeling of fMRI data

---

This is the code source for capstone project of DSI Spring 2023.

There has been high variation and inconsistency between the resting-state functional connectivity (RSFC) and cognitive performance in healthy due to limited sample size and high computational cost for estimation. The Bayesian approach was able to detect a significant diagnostic difference in the association in ROI pairs\cite{wang2020bayesian}. However, implementing MCMC for parameter estimation of Bayesian model in high dimensional spatio-temporal datasets can be difficult due to several reasons. One of the difficulty is the number of possible combinations of parameters grows exponentially in high dimensional case, making it computationally intractable to explore the entire parameter space. As a result, the exploration of the parameter space can become very slow, and it becomes challenging to achieve convergence with traditional MCMC methods. Moreover, some scaling out approach by parallel computing strategy such as subsampling dataset is not suitable in this case due to the charactersitics perceived in spatio-temporal features. Saving computational cost by decreasing number of dimension in sparse dataset and using adaptive MCMC can make algorithm to work more efficiently\cite{robert2018accelerating}.

## Table of Contents

- [Introduction](#introduction)
- [Usage](#usage)
- [Prerequisite packages](#packages)

## Introduction
There has been high variation and inconsistency between the resting-state functional connectivity (RSFC) and cognitive performance in healthy due to limited sample size and high computational cost for estimation. The Bayesian approach was able to detect a significant diagnostic difference in the association in ROI pairs. However, implementing MCMC for parameter estimation of Bayesian model in high dimensional spatio-temporal datasets can be difficult due to several reasons. One of the difficulty is the number of possible combinations of parameters grows exponentially in high dimensional case, making it computationally intractable to explore the entire parameter space. As a result, the exploration of the parameter space can become very slow,
and it becomes challenging to achieve convergence with traditional MCMC methods. Moreover, some scaling out approach by parallel computing strategy such as subsampling dataset is not suitable in this case due to the charactersitics perceived in spatio-temporal features. Saving computational cost by decreasing number of dimension in sparse dataset and using adaptive MCMC can make algorithm to work more efficiently.

## Usage

### [model.py](https://github.com/yutingmeivu/acceleration_fmri/blob/main/PyMC3_model_data/model.py)
  - The modified version of Bayesian spatiotemporal hierarchical model built from previous work with built in Gaussian Processes, jax sampling and GPU.

### [dist_matrix_dim.py](https://github.com/yutingmeivu/acceleration_fmri/blob/main/PyMC3_model_data/dist_matrix_dim.py)
  - Implementation of dimension reduction and visualization of spatial dataset 
  
### [time_matrix_dim.py](https://github.com/yutingmeivu/acceleration_fmri/blob/main/PyMC3_model_data/time_matrix_dim.py)
  - Implementation of dimension reduction and visualization of temporal dataset 
  
## packages
To successfully running the code, the following code is to install the prerequiste pakcages:
```python
pip install jax[cuda11_cudnn82] -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html jaxlib numpyro "pymc>=4" GPy tensorflow sklearn seaborn
```


