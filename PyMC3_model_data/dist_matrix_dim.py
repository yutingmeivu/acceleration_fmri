import tensorflow as tf
from tensorflow.python.client import device_lib

# Check if GPU is available
print(f'has {len(device_lib.list_local_devices())} GPUs')

# Set TensorFlow to use GPU
# tf.config.set_visible_devices(tf.config.list_physical_devices('GPU')[0], 'GPU')


import os
cur_path = "/workspace/testfolder/fmri/PyMC3_model_data/"
os.chdir(cur_path)

import numpy as np
import pandas as pd
import GPy
import multiprocessing as mp

from numpy.core.shape_base import atleast_3d
import pandas as pd
import numpy as np
# import pymc3 as pm
# import pymc as pm
import csv
# import os
import timeit
from datetime import date
from pytensor import *
import pytensor.tensor as at
# from aesara import *
# import aesara.tensor as aa
# from aesara import shared
import pymc.sampling_jax as pmjax
# from pymc.backends import numpyro
# from pymc.inference import _prepare_chain_state
# from numpyro.infer.mcmc import HMCState

import warnings
warnings.filterwarnings('ignore')

import jax.numpy as jnp
from jax import jit, grad
from jax import random
import numpy as np
import GPy
from GPy.models import GPLVM
from sklearn.decomposition import PCA
from scipy.stats import invgamma
from functools import partial
import pymc as pm

os.environ['CUDA_VISIBLE_DEVICES'] = '0,1,2,3'
os.environ['XLA_FLAGS'] = '--xla_dump_to=/tmp/foo'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '0'

physical_devices = tf.config.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(physical_devices[0], True)

############################################################
# this version uses the dynamic mean functional connectivity as input
############################################################

def get_data(name, subID):
    yreader = csv.reader(open(name + "_" + str(subID) + ".csv"))
    Y = np.array([row for row in yreader]).astype(float)
    return Y

def get_func(filename, subID, n):
    Subjectmfunc = filename[filename[:,0] == subID]
    func_new = np.array(Subjectmfunc[0, 1:((n*(n-1)//2)+1)])
    func_temp = np.triu(np.ones([n, n]),1)
    func_temp[func_temp==1] = func_new
    Func_mat = func_temp.T + np.eye(n) + func_temp
    return Func_mat

def get_struct(filename, subID, n):
    SubjectSC = filename[filename[:,0] == subID]
    struct_new = np.array(SubjectSC[0, 1:((n*(n-1)//2)+1)])
    Struct_temp = np.triu(np.ones([n, n]), 1)
    Struct_temp[Struct_temp ==1] = struct_new
    Struct_mat = Struct_temp.T + np.eye(n) + Struct_temp
    return Struct_mat

def get_dist(name, n):
    Dist = []
    num_voxels = []
    sreader = csv.reader(open("AAL_ROI_number.csv"))
    ROI_number = np.array([row for row in sreader]).astype(int)
    for i in range(n):
        distreader = csv.reader(open(name + "_" + str(ROI_number[i,0]) + ".csv"))
        dist_temp = np.array([row for row in distreader]).astype(float)
        Dist.append(dist_temp)
        num_voxels.append(len(dist_temp))
    return Dist, num_voxels


import numpy as np
import pandas as pd
import multiprocessing as mp
import tensorflow as tf
import os


import seaborn as sns
import matplotlib.pyplot as plt
import multiprocessing as mp
import time
from sklearn.preprocessing import StandardScaler

def trans_scalor(x):
    Y_ = x.copy()
    scaler = StandardScaler()
    for i in range(len(x)):
        Y_[i] = scaler.fit_transform(x[i])
    return Y_

def gplvm_dimrec_each_pymc(Y, q, kernel, sample_size, tune_size, alpha, beta):
    
    # Set the range of possible latent dimensions to try
    #min_q = 1
    #max_q = Y.shape[1] - 1

    # Compute the log-likelihood, AIC, and BIC for each value of q
    log_likelihoods = {}
    AIC_values = {}
    BIC_values = {}
    
    with pm.Model() as model:
        if kernel == 'exponential':
            ls_prior = pm.InverseGamma('ls_prior', alpha=alpha, beta=beta)
            cov = pm.gp.cov.ExpQuad(input_dim=q, ls=ls_prior)
        elif kernel == 'periodic':
            period_prior = pm.Gamma('period_prior', alpha=alpha, beta=beta)
            cov = pm.gp.cov.Periodic(input_dim=q, period=period_prior)
        elif kernel == 'matern':
            ls_prior = pm.Gamma('ls_prior', alpha=2, beta=1)
            cov = pm.gp.cov.Matern52(1, ls=ls_prior)
        
        gp = pm.gp.Latent(cov_func=cov)
        f = gp.prior('f', X=np.random.randn(Y.shape[0], q))
        Y_obs = pm.Normal('Y_obs', mu=f, sigma=0.1, observed=Y.T)

        # Sample the posterior distribution of latent space using NUTS
        #trace = pm.sample(draws=1000, tune=1000, chains=2, cores=2)
        trace = pmjax.sample_numpyro_nuts(draws=sample_size, tune=tune_size,
                postprocessing_backend="gpu", chain_method="vectorized",
                chains=1, idata_kwargs={"log_likelihood": True})
        
    log_likelihood = trace.log_likelihood.y  
    num_params = len(trace[-1])
    AIC_values = 2*num_params - 2*log_likelihood
    BIC_values = np.log(N)*num_params - 2*log_likelihood

#         if multi:
#             for q in range(min_q, max_q):
#                 log_likelihood = pm.gp.util.get_log_likelihood(model, point=trace[-1], include_obs=True)
#                 log_likelihoods[q] = log_likelihood
#                 num_params = len(trace[-1])
#                 N = Y.shape[0]
#                 AIC_values['AIC'+ str(q)] = 2*num_params - 2*log_likelihood
#                 BIC_values['BIC' + str(q)] = np.log(N)*num_params - 2*log_likelihood

    return (AIC_values, BIC_values, trace)

def gplvm_dimrec_each_gpy(Y, q, plot, kern, alpha, beta, path, i):
    
    # Compute the log-likelihood, AIC, and BIC for each value of q
    log_likelihoods = {}
    AIC_values = {}
    BIC_values = {}
    #     for q in range(min_q, max_q):
    if kern == 'exponential':
        #ls_prior = invgamma(alpha, scale=beta)
        ls_prior = invgamma.rvs(a=3, scale=0.5, size=1)[0]
        k = GPy.kern.Exponential(input_dim=q, lengthscale=ls_prior)
    elif kern == 'periodic':
        period_prior = GPy.priors.Gamma(1.0, 1.0)
        k = GPy.kern.PeriodicMatern32(input_dim=q, prior=period_prior)
        #     m = GPLVM(Y, input_dim=q, kernel=k, X=np.random.randn(Y.shape[0], q)) # Random initialization
    elif kern == 'matern':
        ls_prior = GPy.priors.Gamma(alpha, beta)
        # Define a Matern kernel with the gamma prior on the lengthscale
        k = GPy.kern.Matern32(input_dim=q, lengthscale=None, ARD=True)
        # Set the gamma prior on the lengthscale parameter
        k.lengthscale.set_prior(ls_prior)
    m = GPLVM(Y = Y, input_dim=q, kernel=k)
    ker = k.copy()
    m.optimize(max_iters=300, messages=True)
    #log_likelihood_fn = partial(log_likelihood(Y, m.optimizer_array), Y=Y)
    #log_likelihood = jit(log_likelihood_fn, static_argnums=(0,))   
    log_likelihood = m.log_likelihood()
    num_params = m.param_array.size
    N = 1000
    AIC_values['AIC'+ str(q)] = 2*num_params - 2*log_likelihood
    BIC_values['BIC' + str(q)] = np.log(N)*num_params - 2*log_likelihood
    if plot:
        m.data_labels = np.arange(Y.shape[0])
        labels=m.data_labels
        fig, ax = m.plot_latent(figsize=(8, 6), cmap='viridis', label = ['GPLVM'], scatter_kwargs={'s': 85, 'color': 'white'})
        plt.legend()
#         pca = PCA(n_components=q)
#         Y_pca = pca.fit_transform(Y)
        # Plot the PCA result
#         plt.scatter(Y_pca[:, 0], Y_pca[:, 1], label='PCA', marker = '+', c = 'white', s = 85)
        plt.title(f'GPLVM with dimension reduction to {q}')
        plt.savefig(path + 'figures/spatial_reduction_gpy_n' + str(q) + '_'+  str(i) + 'index' +'.png')
        
#         plt.legend()
        # Plot the GPLVM result
#         m.plot_latent(labels=m.data_labels, cmap='flare')
#         plt.title(f'GPLVM with latent space {q}')
#         plt.legend()
#         plt.savefig(path + 'figures/time_rec' + str(q) + '.png')
        plt.show()
    d_temp = {'AIC': AIC_values['AIC'+ str(q)], 'BIC': BIC_values['BIC'+ str(q)]}
    dff = pd.DataFrame(d_temp, index = [0])
    dff.to_csv(hpath + 'result/info_value_spatial'+str(q) + '_' + 'number_id' + str(i) +'.csv')
    return (AIC_values, BIC_values)


def get_info(info_result):
    global info_pymc
    info_pymc.append(info_result)

def multi_dim_gpy(q_min, q_max, num_voxel, dt, path):
    AIC_sum = {}
    BIC_sum = {}
    trace_sum = {}
    pool = mp.Pool(256)
    for i in range(num_voxel):
        
        if __name__ == '__main__':
            info_pymc = []
            pool.starmap_async(gplvm_dimrec_each_gpy, [(np.array(dt[i]), q, True, 'matern', 2, 0.8, path, i) for q in range(q_min, min(q_max, dt[i].shape[0]), 15)], \
                                     callback = get_info).get()
#             pool.close()
#             AIC_dim['dim_'+ str(i)] = info_pymc[-3]
#             BIC_dim['dim_'+ str(i)] = info_pymc[-2]
#             trace_sum['dim_'+str(i)] = info_pymc[-1]
#             del info_pymc
#             AIC_df = pd.DataFrame(AIC_dim)
#             BIC_df = pd.DataFrame(BIC_dim)
#             trace_df = pd.DataFrame(trace_sum)
#             AIC_df.to_csv(path + 'AIC' + i + '_dist.csv')
#             BIC_df.to_csv(path + 'BIC' + i + '_dist.csv')
#             trace_df.to_csv(path + 'trace' + i+ '_dist.csv')
#             del AIC_dim
#             del BIC_dim
#             del info_pymc
#             AIC_dim = {}
#             BIC_dim = {}
            
            
hpath = '/workspace/testfolder/fmri/PyMC3_model_data/'

os.chdir(hpath)

path = hpath + 'distance_matrix/'
os.chdir(path)
n = 90
save_path = hpath + 'result/'
dist_filename = "distance_matrix"
Dist, num_voxels = get_dist(dist_filename, n)


Dist_ = Dist.copy()


Dist_ = trans_scalor(Dist_)

start = time.time()

multi_dim_gpy(5, 250, n, Dist_, hpath)

end_time = time.time() - start

print(f"Finish running of {index} with {end_time} seconds.")

