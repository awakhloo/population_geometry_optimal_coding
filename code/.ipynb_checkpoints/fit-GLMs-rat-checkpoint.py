#!/usr/bin/env python

#SBATCH --time=1-00:00:00
#SBATCH --partition ccn
#SBATCH --nodes=1
#SBATCH -C icelake

import sys
import os 
sys.path.append(os.getcwd()) 

import numpy as np 
import matplotlib.pyplot as plt 
from glob import glob
import pandas as pd
from functools import partial
from sklearn.linear_model import PoissonRegressor
from sklearn.multioutput import MultiOutputRegressor
from sklearn.model_selection import cross_val_score, GridSearchCV
from mpl_toolkits.axes_grid1 import make_axes_locatable
from tqdm import tqdm
import rat_utils as rat 

import warnings
from sklearn.exceptions import ConvergenceWarning
import warnings
warnings.filterwarnings("ignore")


dt = 0.5 # spike bin size in seconds
vmin=4 # minimum velocity to consider in cm/s 
n_rat = 8 # num rats and sessions
n_sesh = 8

outdir = '/mnt/home/awakhloo/ceph/reprod_population_geom_opt_coding/results'
paths = glob(outdir+ f'/processed_rat_dat/{str(dt)}/*.npy')

def refit_strategy(cv_results, cv):
    ''' 
    instructions for choosing best model following grid search.
    ignore nans when taking the mean over splits, which can occur for sparse neurons during cross validation 
    (specifically when the variance of the hold-out set is 0) 
    '''
    scores = np.stack([cv_results[f'split{i}_test_score'] for i in range(cv)], axis=0)
    scores = np.nanmean(scores,axis=0) 
    return np.argmax(scores)

def get_test_scores(mod, cv): 
    ''' 
    avg mean and st dev of the score across cross val splits
    return also the number of valid cross val trials per neuron 
    '''
    all_means = [] 
    all_devs = [] 
    all_num_valid = [] 
    for est in model.estimators_:
        scores = np.stack([est.cv_results_[f'split{i}_test_score'] for i in range(cv)], axis=0) # a vector of shape n_params
        means = np.nanmean(scores,axis=0) 
        devs = np.nanstd(scores,axis=0)
        num_valid = np.sum(~np.isnan(scores),axis=0) # number of cross val splits that allowed for the model to be fit in the first place
        all_means.append(means)
        all_devs.append(devs)
        all_num_valid.append(num_valid) 
    return np.stack(all_means), np.stack(all_devs), np.stack(all_num_valid) # n_neur x n_param 




area = 'CA1' ##### SET TO 'PFC' or 'CA1' DEPENDING ON WHICH AREA YOU WANT TO ANALYZE!

bin_sizes = [5, 10, 15] # bin size space 
bin_sizes_vel = [3, 6, 12] # bin size velocity
ext_sps = [10, 50, 100, 150] # variance of spatial gaussian bases
ext_svs = [10, 40, 80] # variance of velocity bases
split_traj = True # separate covariates by trajectory type 
xmin = 0.1 # firing rate cutoff in Hz 
min_count = 20 # minimum number of trials to consider including a bin 


param_grid = {'alpha' : np.logspace(-1/2, 2, 10)}
cv = 10 # number of cross-val folds 


all_z, all_x, flat = rat.get_all_lat(paths, area=area, vmin=vmin, n_sesh=n_sesh)


glm_path = outdir + '/glm_results'
os.makedirs(glm_path, exist_ok=True)
for bin_size in bin_sizes: 
    for bin_size_vel in bin_sizes_vel: 
        for sp in ext_sps: 
            for sv in ext_svs: 
                all_results = [] 
                # make the covariates
                for i in tqdm(range(n_rat)): 
                    # bin space and get basis functions 
                    posn = flat[i][:, :2] 
                    bins = rat.bin_positions(posn, bin_size, min_count) 
                    basis_fns_posn = [partial(rat.iso_gauss, m=b, s=sp) for b in bins]
                    # bin velocity 
                    vels = flat[i][:, 3:5]
                    bins_vx, bins_vy = rat.bin_velocities(vels[:,0], bin_size_vel, min_count), rat.bin_velocities(vels[:,1], bin_size_vel, min_count)
                    basis_fns_vx, basis_fns_vy = [partial(rat.gauss_basis, m=b, s=sv) for b in bins_vx], [partial(rat.gauss_basis, m=b, s=sv) for b in bins_vy]
                    # build up session by session covariates 
                    this_rat = [] 
                    for s in tqdm(range(n_sesh)): 
                        z = all_z[i][s] 
                        covs = rat.get_covs(z, basis_fns_posn, basis_fns_vx, basis_fns_vy,split_traj=split_traj)
                        covs = (covs - covs.mean(0))/covs.std(0)
                        x = all_x[i][s] 
                        # drop dead units 
                        x = x[:, x.mean(0)/dt > xmin]
                
                        ### MODEL FITTING 
                        reg=PoissonRegressor()
                        search=GridSearchCV(reg, param_grid, n_jobs=-1, cv=cv, error_score='raise', refit=partial(refit_strategy, cv=cv))
                        model=MultiOutputRegressor(search,n_jobs=-1)
                        model.fit(covs, x)
                        
                        # results 
                        alphas = np.array([model.estimators_[i].best_estimator_.get_params()['alpha'] for i in range(x.shape[1])])
                        test_scores, test_devs, num_valid = get_test_scores(model, cv=cv)
                        xhat = model.predict(covs)
                        results = {'x' : x, 
                                   'z' : z, 
                                   'covs' : covs,
                                   'xhat' : xhat,
                                   'alphas' : alphas, 
                                   'test_scores' : test_scores,
                                  'test_devs' : test_devs, 
                                   'num_valid' : num_valid, 
                                  'glm_params' : [bin_size, bin_size_vel, sp, sv, min_count, xmin, cv]}
                
                        this_rat.append(results)
                    all_results.append(this_rat)
                np.save(glm_path +f'/glm_res_{area}_bin_size_{bin_size}_bin_size_vel_{bin_size_vel}_sp_{sp}_sv_{sv}.npy', 
                        np.array({'results' : all_results}))
            
            
            