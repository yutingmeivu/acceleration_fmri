from numpy.core.shape_base import atleast_3d
import pandas as pd
import numpy as np
# import pymc3 as pm
import pymc as pm
# import theano
# import theano.tensor as tt
# from theano import printing
import csv
import os
import timeit
from datetime import date
from pytensor import *
import pytensor.tensor as at
import arviz as az
import tensorflow as tf
#from aesara import *
#import aesara.tensor as aa
#from aesara import shared
import pymc.sampling_jax as pmjax
# os.environ['XLA_PYTHON_CLIENT_PREALLOCATE']='false'
os.environ['CUDA_VISIBLE_DEVICES'] = '1,2,3'
os.environ['XLA_FLAGS'] = '--xla_dump_to=/tmp/foo'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '0'
physical_devices = tf.config.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(physical_devices[0], True)
 
import warnings
warnings.filterwarnings('ignore')

############################################################
# this version uses the dynamic mean functional connectivity as input
# ###########################################################

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
 
def run_model(index, out_dir, Y, mFunc, Struct, Dist, n, kernel,  sample_size, tune_size, num_voxels):
    """
    index: index of object data
    total_voxel: a total number of voxels in all ROIs
    Y: time-series data
    mFunc: functional connectivity
    Struct: structural connectivity
    Dist: distance matrix of n ROIs
    n: ROI number
    kernel: "exponential" or "gaussian" or "matern52" or "matern32"
    lambdaw: weighted parameter
    kf: weighted parameter
    sample_size: NUTS number
    tune_size: burning number
    """
    k = Y.shape[1]
    n_vec= n*(n+1)//2
    total_voxel = sum(num_voxels)
   
    num_cum_sum = np.append(0, np.cumsum(num_voxels))
    Y_mean = []
    for i in range(n):
        Y_mean.append(np.mean(Y[num_cum_sum[i]:num_cum_sum[i+1], 0]))
    Y_mean = np.array(Y_mean)
   
    with pm.Model() as model_generator:
       
        # convariance matrix
        log_Sig = pm.Uniform("log_Sig", -8, 8, shape = (n, ))
        #log_Sig = np.reshape(log_Sig, (-1, ))
        #log_Sig = [i for i in log_Sig]
        # print(list(np.sqrt(np.exp(log_Sig))))
        SQ = at.diag(at.sqrt(at.exp(log_Sig)))
        Func_Covm = at.dot(at.dot(SQ, mFunc), SQ)
        Struct_Convm = at.dot(at.dot(SQ, Struct), SQ)
        lambdaw = pm.Beta("lambdaw", alpha = 1, beta = 1, shape = (n_vec, ))
        Kf = pm.Beta("Kf", alpha = 1, beta = 1, shape = (n_vec, ))
       
        # double fusion of structural and FC
        L_fc_vec = at.reshape(at.slinalg.cholesky(at.squeeze(Func_Covm)).T[at.triu_indices(n)], (n_vec, ))
        L_st_vec = at.reshape(at.slinalg.cholesky(at.squeeze(Struct_Convm)).T[at.triu_indices(n)], (n_vec, ))
        # print([at.triu_indices(n)])
        Struct = pm.Data('Struct', Struct, shape = Struct.shape)
        Struct_vec = at.reshape(Struct[at.triu_indices(n)], (n_vec, ))
        rhonn = Kf*( (1-lambdaw)*L_fc_vec + lambdaw*L_st_vec ) + \
            (1-Kf)*( (1-Struct_vec*lambdaw)*L_fc_vec + Struct_vec*lambdaw*L_st_vec )
       
        # correlation
        Cov_temp = at.triu(at.ones((n,n)))
        Cov_temp = at.set_subtensor(Cov_temp[at.triu_indices(n)], rhonn)
        Cov_mat_v = at.dot(Cov_temp.T, Cov_temp)
        d = at.sqrt(at.diagonal(Cov_mat_v))
        rho = (Cov_mat_v.T/d).T/d
        rhoNew = pm.Deterministic("rhoNew", rho[np.triu_indices(n,1)])
      
        # temporal correlation AR(1)
        phi_T = pm.Uniform("phi_T", 0, 1, shape = (n, ))
        sigW_T = pm.Uniform("sigW_T", 0, 100, shape = (n, ))
        B = pm.Normal("B", 0, 100, shape = (n, ))
        muW1 = Y_mean - B # get the shifted mean
        mean_overall = muW1/(1.0-phi_T) # AR(1) mean
        tau_overall = (1.0-at.sqr(phi_T))/at.sqr(sigW_T) # AR (1) variance
        L_overall = at.linalg.cholesky(at.diag(tau_overall))
        W_T = pm.MvNormal("W_T", mu=mean_overall, chol=L_overall, shape=(k, n))
        # W_T = pm.MvNormal("W_T", mu = mean_overall, tau = at.diag(tau_overall), shape = (k, n))
        # L_overall = at.slinalg.cholesky(tau_overall)
        # W_T = pm.Deterministic("W_T", mean_overall + at.dot(L_overall, pm.Normal.dist(0, 1, shape=(k, n)).draw()))
        # norm_samples = pm.Normal("norm_samples", mu=0, sigma=1, shape=(k, n))
        # W_T = pm.Deterministic("W_T", mean_overall + at.dot(L_overall, norm_samples.T))


       
        # add all parts together
       
        one_k_vec = at.ones((1, k))
        Cov_mat_v_L = at.linalg.cholesky(Cov_mat_v)
        D = pm.MvNormal("D", mu = at.zeros(n), chol = Cov_mat_v_L, shape = (n, ))
        phi_s = pm.Uniform("phi_s", 0, 20, shape = (n, ))
        spat_prec = pm.Uniform("spat_prec", 0, 100, shape = (n, ))
        
        kernel = pm.gp.cov.ExpQuad(1, ls=phi_s)
        gp = pm.gp.Marginal(cov_func=kernel)
        

       
        
        Mu_all = at.zeros((total_voxel, k))
       
        #printing_op = printing.Print('vector', attrs = [ 'shape' ])
       
        if kernel == "exponential":
            # for i in range(n):
            #     one_m_vec = at.ones((num_voxels[i], 1))
            #     H_base = pm.Normal("H_base_" + str(i), 0, 1, shape = (num_voxels[i], 1))
            #     r = Dist[i]*phi_s[i]
            #     H_temp = at.sqr(spat_prec[i])*at.exp(-r)
            #     L_H_temp = at.slinalg.cholesky(H_temp)
            #     print(i, num_voxels[i])
            #     #printing_op(L_H_temp)
            #     #printing_op(H_base)
            #     print(num_cum_sum[i], num_cum_sum[i+1])
            #     # print(Mu_all)
            #     Mu_all_update = at.set_subtensor(Mu_all[num_cum_sum[i]:num_cum_sum[i+1], :], B[i] + D[i] + one_m_vec*W_T[:,i] + \
            #             at.dot(L_H_temp, H_base)*one_k_vec)
            #     Mu_all = Mu_all_update
            #     del H_base
            #     # del(H_base_)
            #     # for name in dir():
            #     #   if name == 'H_base':
            #     #     del globals()[name]  
            for i in range(n):
                one_m_vec = at.ones((num_voxels[i], 1))
                H_base = pm.Normal("H_base_" + str(i), 0, 1, shape = (num_voxels[i], 1))
                r = Dist[i] * phi_s[i]
                H_temp = at.sqr(spat_prec[i]) * at.exp(-r)
                L_H_temp = at.slinalg.cholesky(H_temp)
                # Define mean function for the Gaussian process
                mu = pm.Normal('mu_' + str(i), mu=0, sigma=1, shape=num_voxels[i])
                # Define the latent Gaussian process
                f = gp.marginal_likelihood('f_' + str(i), X=Dist[i], y=mu, noise=spat_prec[i] ** 2)
                Mu_all_update = at.set_subtensor(
                    Mu_all[num_cum_sum[i]:num_cum_sum[i+1], :],
                    B[i] + D[i] + one_m_vec * f.T + at.dot(L_H_temp, H_base) * one_k_vec)
                Mu_all = Mu_all_update
            
            # n = len(Dist)
#             num_vo = [i.shape[0] for i in Dist]
#             Dist_shared = np.empty((max(num_vo), max(num_vo), n), dtype=float)
#             for i in range(n):
#                 Dist_shared[:num_vo[i], :num_vo[i], i] = Dist[i]
#             Dist_shared = pm.floatX(Dist_shared)

                
#             kernel = pm.gp.cov.ExpQuad(1, ls=phi_s)
#             # define latent Gaussian process
#             gp = pm.gp.Latent(cov_func=kernel)
#             # define mean function for the Gaussian process
#             one_m_vec = at.ones((sum(num_voxels), 1))
#             mean = at.dot(W_T.T, one_m_vec)
#             mean_reshaped = at.tile(mean, (Dist_shared.shape[0], Dist_shared.shape[1], 1))
#             print(mean_reshaped.shape)
#             # obtain the latent function by adding the mean function to the Gaussian process
#             f = gp.prior("f", X=Dist_shared) + mean_reshaped
#             # set the dimensions of the output matrix
#             Mu_all = pm.Deterministic("Mu_all", f.reshape(sum(num_voxels), k))

                
            
#             kernel = pm.gp.cov.ExpQuad(1, ls=phi_s)
#             spatial = pm.gp.Marginal(cov_func=kernel)
            
#             mu_b = pm.Normal("mu_b", mu=0, sigma=1, shape=(n, 1))
#             sigma_b = pm.Uniform("sigma_b", lower=0, upper=10, shape=(n, 1))
#             B = pm.Normal("B", mu=mu_b, sigma=sigma_b, shape=(n, num_voxels.max()))
            
#             W_T = pm.gp.Latent(cov_func=kernel)
#             f_T = W_T.prior("f_T", X=Dist_shared)
#             B_D = at.concatenate([(B[i] + D[i]).reshape(-1, 1) for i in range(n)], axis=0)
#             Mu_all_update = at.set_subtensor(
#                 Mu_all,
#                 B_D + one_m_vec * f_T.T + one_k_vec * at.zeros((k, num_voxels.sum())))
#             Mu_all = Mu_all_update

            

        elif kernel == "gaussian":
            for i in range(n):
                one_m_vec = np.ones((num_voxels[i], 1))
                H_base = pm.Normal("H_base", 0, 1, shape = (num_voxels[i], 1))
                r = Dist[i]*phi_s[i]
                H_temp = at.sqr(spat_prec[i])*at.exp(-np.sqr(r)*0.5)
                L_H_temp = at.slinalg.cholesky(H_temp)
                Mu_all_update = at.set_subtensor(Mu_all[num_cum_sum[i]:num_cum_sum[i+1]-1, :], B[i] + D[i] + one_m_vec*W_T[:,i] + \
                    at.dot(L_H_temp, at.reshape(H_base, (num_voxels[i], 1)))*one_k_vec)
                Mu_all = Mu_all_update
        elif kernel == "matern52":
            for i in range(n):
                one_m_vec = at.ones((num_voxels[i], 1))
                H_base = pm.Normal("H_base", 0, 1, shape = (num_voxels[i], 1))
                r = Dist[i]*phi_s[i]
                H_temp = at.sqr(spat_prec[i])*((1.0+at.sqrt(5.0)*r+5.0/3.0*at.sqr(r))*at.exp(-1.0*at.sqrt(5.0)*r))
                L_H_temp = at.slinalg.cholesky(H_temp)
                Mu_all_update = at.set_subtensor(Mu_all[num_cum_sum[i]:num_cum_sum[i+1]-1, :], B[i] + D[i] + one_m_vec*W_T[:,i] + \
                    at.dot(L_H_temp, at.reshape(H_base, (num_voxels[i], 1)))*one_k_vec)
                Mu_all = Mu_all_update
        elif kernel == "matern32":
            for i in range(n):
                one_m_vec = np.ones((num_voxels[i], 1))
                H_base = pm.Normal("H_base", 0, 1, shape = (num_voxels[i], 1))
                r = Dist[i]*phi_s[i]
                H_temp = at.sqr(spat_prec[i])*(1.0+at.sqrt(3.0)*r)*at.exp(-at.sqrt(3.0)*r)
                L_H_temp = at.slinalg.cholesky(H_temp)
                Mu_all_update = atleast_3d.set_subtensor(Mu_all[num_cum_sum[i]:num_cum_sum[i+1]-1, :], B[i] + D[i] + one_m_vec*W_T[:,i] + \
                    at.dot(L_H_temp, at.reshape(H_base, (num_voxels[i], 1)))*one_k_vec)
                Mu_all = Mu_all_update
       
        sigma_error_prec = pm.Uniform("sigma_error_prec", 0, 100)
        Y1 = pm.Normal("Y1", mu = Mu_all, sigma = sigma_error_prec, observed = Y)
   
    with model_generator:
        #step = pm.NUTS()
        #trace = pm.sample(sample_size, step = step, tune = tune_size, chains = 1)
        trace = pmjax.sample_numpyro_nuts(draws=sample_size, tune=tune_size,
                postprocessing_backend="cpu", chain_method="vectorized",
                chains=1, idata_kwargs={"log_likelihood": False})
 
    # save as nc file in Arviz
    trace.to_netcdf(out_dir + date.today().strftime("%m_%d_%y") + \
        "_sample_size_" + str(sample_size) + "_index_" + str(index) +".nc")
# initializing parameters
#index_list = [8007, 8012, 8049, 8050, 8068, 8072, 8077, 8080, \
#              8098, 8107, 8110, 8146, 8216, 8244, 8245, 8246, \
#              8248, 8250, 8253, 8256, 8257, 8261, 8262, 8263, \
##              8264, 8265, 8266, 8273, 8276, 8279, 8280, 8282, \
#              8283, 8284, 8285, 8288, 8290, 8292, 8293, 8295, \
#              8299]
 
def fin_run_model(index, root_dir):
    # in_dir = root_dir + "/data/YoungAdults/" # in_dir: set up work directory
    in_dir = root_dir
    # out_dir = root_dir + "/results/YoungAdults/Static_FC/" # out_dir: save the trace as csv in the out directory
    out_dir = root_dir + '/results/'
    data_filename = "Resting_fMRI_doubleFusion" # data_filename: filename for time series data
    # func_filename = "DMN_MeanFunctional_Connectivity" # func_filename: filename for functional connectivity
    # struct_filename = "DMN_Structural_Connectivity" # struct_filename: filename for structural connectivity
    dist_filename = "distance_matrix" # dist_filename: filename for distribution matrix of n ROIs
 
    kernel = "exponential"
    n = 90
    sample_size = 1000
    tune_size = 1000 ## make it smaller for dynamic FC
 
    # run the model
    # for index in index:
    # obtain the previous results
    # df_previous = pd.read_csv(in_dir + "/all_generating_data/sample_size_" + str(sample_size) + "_index_" + str(index) + ".csv")
    # lambdaw estimated from the median of previsous results, (105, )
    # lambdaw = np.squeeze(df_previous[df_previous.columns[df_previous.columns.str.startswith("lambdaw")]].median(axis = 0).values)
    # Kf estimated from the median of previsous results, (105, )
    # Kf = np.squeeze(df_previous[df_previous.columns[df_previous.columns.str.startswith("Kf")]].median(axis = 0).values)
    # phi_s = np.squeeze(df_previous[df_previous.columns[df_previous.columns.str.startswith("phi_s")]].median(axis = 0).values)
    # spat_prec = np.squeeze(df_previous[df_previous.columns[df_previous.columns.str.startswith("spat_prec")]].median(axis = 0).values)
    # H_base = np.squeeze(df_previous[df_previous.columns[df_previous.columns.str.startswith("H_base")]].median(axis = 0).values)
    os.chdir(in_dir + "/distance_matrix/")
    Dist, num_voxels = get_dist(dist_filename, n)
   
    # os.chdir(in_dir + "/rs_fMRI_data/")
    os.chdir(in_dir)
    Y = get_data(in_dir + '/' + data_filename, index)
   
    Msreader = csv.reader(open("/workspace/testfolder/fmri/PyMC3_model_data/Conventional_FC.csv"))
    mFuncinput = np.array([row for row in Msreader]).astype(float)
    mFunc = get_func(mFuncinput, index, n)
   
    SCsreader = csv.reader(open("/workspace/testfolder/fmri/PyMC3_model_data/SC_matrix.csv"))
    SCinput = np.array([row for row in SCsreader]).astype(float)
    Struct = get_struct(SCinput, index, n)
   
    # t_total = Y.shape[1] # the total is 150
    # for slide_index in range(t_total-t_interval):
    run_model(index, out_dir, Y, mFunc, Struct, Dist, n, kernel,  sample_size, tune_size, num_voxels)

import time
import multiprocessing as mp
# from numba import jit, cuda
#start_time = time.time()
hpath = '/workspace/testfolder/fmri/PyMC3_model_data'
os.chdir(hpath)
path = os.getcwd()
# @jit(target_backend='cuda')
def run(index, path):
    fin_run_model(index, path)

start_time = time.time()
path = os. getcwd()
run(8007, path)
print("--- %s seconds ---" % (time.time() - start_time))
#pool = mp.Pool(10)
#pool.apply_async(fin_run_model, args = (8007, path))
#print("--- %s seconds ---" % (time.time() - start_time))
#pool.close()
#pool.join()
