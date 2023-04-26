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

os.environ['CUDA_VISIBLE_DEVICES'] = '0,1,2,3'
os.environ['XLA_FLAGS'] = '--xla_dump_to=/tmp/foo'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

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

def gplvm_dimrec_each(X, q):

    # Set the range of possible latent dimensions to try
    min_q = 1
    max_q = X.shape[1] - 1

    # Compute the log-likelihood, AIC, and BIC for each value of q
    log_likelihoods = {}
    AIC_values = {}
    BIC_values = {}
    # for q in range(min_q, max_q):
    m = GPy.models.GPLVM(X, input_dim=q, kernel=GPy.kern.RBF(q, ARD=True))
    m.optimize()
    log_likelihoods[q] = m.log_likelihood()
    num_params = m.param_array.size
    N = X.shape[0]
    AIC_values['AIC'+ str(q)] = 2*num_params - 2*log_likelihoods[q]
    BIC_values['BIC' + str(q)] = np.log(N)*num_params - 2*log_likelihoods[q]

    return (AIC_values, BIC_values)


def multi_gplvm_hybrid(Y, q_min, q_max, path):
    
    # Initialize a TensorFlow session to use the GPU
    # gpu_options = tf.GPUOptions(allow_growth=True)
    # config = tf.ConfigProto(gpu_options=gpu_options)
    # sess = tf.Session(config=config)

    physical_devices = tf.config.list_physical_devices('GPU')
    tf.config.experimental.set_memory_growth(physical_devices[0], True)

    # Create a pool of worker processes to parallelize the computations
    pool = mp.Pool(100)
    results = pool.starmap(gplvm_dimrec_each, [(Y, q) for q in range(q_min, q_max)]).get()
    pool.close()

    # Merge the results into a global dataframe and write it to disk
    AIC = pd.DataFrame(list(map(lambda x: x[0], results)))
    BIC = pd.DataFrame(list(map(lambda x: x[1], results)))
    AIC.to_csv(path + 'AIC_hybrid.csv')
    BIC.to_csv(path + 'BIC_hybrid.csv')
    
    # Terminate the TensorFlow session to free up GPU memory
    # sess.close()

    return (AIC, BIC)


def trans_scalor(x):
    Y_ = x.copy()
    scaler = StandardScaler()
    for i in range(len(x)):
        Y_[i] = scaler.fit_transform(x[i])
    return Y_


def dist_multi(Y_, i, q_min, q_max):
    aic_dim = {}
    bic_dim = {}
    aic = []
    bic = []
    # for i in range(Y_.shape[0]):
    for j in range(q_min,q_max):
        AIC_values, BIC_values = gplvm_dimrec_each(Y_[i], j)
        aic.append(AIC_values)
        bic.append(BIC_values)
        aic_dim['dim_'+ str(i)] = aic
        bic_dim['dim_'+ str(i)] = bic
        aic = []
        bic = []
    return (aic_dim, bic_dim)


def latent_reduce_dim_seperate(Y, q_min, q_max, index, path):
    # index: roi index
    physical_devices = tf.config.list_physical_devices('GPU')
    tf.config.experimental.set_memory_growth(physical_devices[0], True)

    pool = mp.Pool(100)

    results = pool.starmap(gplvm_dimrec_each, [(Y[index], q) for q in range(q_min, q_max)]).get()

    pool.close()

    AIC = pd.DataFrame(list(map(lambda x: x[0], results)))
    BIC = pd.DataFrame(list(map(lambda x: x[1], results)))
    AIC.to_csv(path + 'AIC_dist_' + index + '.csv')
    BIC.to_csv(path + 'BIC_dist_' + index + '.csv')
    return (AIC, BIC)


ftime = pd.read_csv("Resting_fMRI_doubleFusion_8007.csv", header=None)

import seaborn as sns
import matplotlib.pyplot as plt
import multiprocessing as mp
import time
from sklearn.preprocessing import StandardScaler


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


key = random.PRNGKey(0)

def log_likelihood(Y, optimizer_array):
    m.optimizer_array = optimizer_array
    m._trigger_params_changed()
    return -m.log_likelihood(Y=Y)

def gplvm_dimrec_each(Y, q, plot, kern, alpha, beta, path):
    
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
    m = GPLVM(Y = Y, input_dim=q, kernel=k)
    ker = k.copy()
    m.optimize(max_iters=75, messages=True)
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
        plt.savefig(path + 'figures/time_reduction_gpy_n' + str(q) + '.png')
        
#         plt.legend()
        # Plot the GPLVM result
#         m.plot_latent(labels=m.data_labels, cmap='flare')
#         plt.title(f'GPLVM with latent space {q}')
#         plt.legend()
#         plt.savefig(path + 'figures/time_rec' + str(q) + '.png')
        plt.show()
    d_temp = {'AIC': AIC_values['AIC'+ str(q)], 'BIC': BIC_values['BIC'+ str(q)]}
    dff = pd.DataFrame(d_temp, index = [0])
    dff.to_csv(hpath + 'result/info_value'+str(q) +'.csv')
    return (AIC_values, BIC_values)


scaler = StandardScaler()
dff = scaler.fit_transform(ftime)
# ld = 85
kern = 'exponential'
hpath = '/workspace/testfolder/fmri/PyMC3_model_data/'

def get_rtime(info_):
    global info_time
    info_time.append(info_)

# info = gplvm_dimrec_each(np.array(dff), ld, True, kern, 1, 1)



# info = gplvm_dimrec_each(np.array(dff), ld, True, kern, 1, 1)

loop = range(5, 149, 15)

pool = mp.Pool(256)
if __name__ == '__main__':
    info_time = []
    pool.starmap_async(gplvm_dimrec_each, [(np.array(dff), q, True, kern, 1, 10, hpath) for q in loop], \
                                     callback = get_rtime).get()
pool.close()

info_time = info_time[0]
# AIC_time = list(map(lambda x: x[0], info_time))
# BIC_time = list(map(lambda x: x[1], info_time))
# AIC_time = pd.DataFrame(AIC_time)
# BIC_time = pd.DataFrame(BIC_time)
# AIC_time.to_csv(hpath + 'AIC_time.csv')
# BIC_time.to_csv(hpath + 'BIC_time.csv')

aic_values = [d[list(d.keys())[0]] for d in [t[0] for t in info_time]]
bic_values = [d[list(d.keys())[0]] for d in [t[1] for t in info_time]]

df = pd.DataFrame({'AIC': aic_values, 'BIC': bic_values})
df['latent_dim'] = [i for i in loop]
print(df)
df.to_csv(hpath + 'result/total_time.csv')
