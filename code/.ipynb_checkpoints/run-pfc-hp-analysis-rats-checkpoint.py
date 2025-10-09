#!/usr/bin/env python

#SBATCH --time=1-00:00:00
#SBATCH --partition ccn
#SBATCH --nodes=1
#SBATCH -C rome


import numpy as np 
from tqdm import tqdm
from glob import glob
import sys
import os 
sys.path.append(os.getcwd()) 
import rat_utils as rat
from utils import theory_part, emp_error, get_geometry 

outdir = '/mnt/home/awakhloo/ceph/reprod_population_geom_opt_coding/results'


def get_learning_curve(xraw, zraw, locs, P=100, subset=None, area='CA1',
                      vmin=None, subset_neur=None,
                      dt=None, xmin=None):
    targs=np.arange(len(xraw))
    emps = [] 
    thes = [] 
    geoms = [] 
    svcs = [] 
    # loop through sessions 
    for targ in targs: 
        xr, zr, loc = xraw[targ], zraw[targ], locs[targ]
        x, z = rat.process_dat(zr,xr, loc, dt=dt, area=area, vmin=vmin, xmin=xmin)  
        
        print ('using location and velocity')
        z = np.concatenate([z[:,:2], z[:, 3:5]], axis=1)

        if subset is not None: 
            inds = np.random.choice(np.arange(x.shape[0]), size=(subset,), replace=False)
            x,z = x[inds], z[inds]

        if subset_neur is not None: 
            neur_inds = np.random.choice(np.arange(x.shape[1]), size=(subset_neur,), replace=False)
            x = x[:, neur_inds] 
            
        print('xshape = ', x.shape, flush=True)
        n,d = x.shape[1], z.shape[1]       
        X = np.concatenate([x,z], axis=1)
        Xhat = (X - X.mean(0,keepdims=True))
        C = Xhat.T @ Xhat / Xhat.shape[0]
        
        Psi, Phi, Omega = C[:n,:n], C[:n, n:], C[n:, n:] 

        emp = [] 
        emp_results = emp_error(x, z, P, svc_analysis=True)
        emp.append(emp_results[0])
        svcs.append(emp_results[1])
        
        geom = get_geometry(Psi, Phi, Omega, P)
        emps.append(np.mean(emp))
        thes.append(theory_part(Psi,Phi,Omega,P))
        geoms.append(np.array(geom)) 
        
    return np.array(emps), np.array(thes), np.stack(geoms), np.array(svcs)



dt = 0.5
paths = glob(outdir+ f'/processed_rat_dat/{str(dt)}/*.npy')

errs = [] 
all_geoms = [] 
all_tdims = [] 

area='CA1' ##### SET TO 'PFC' or 'CA1' DEPENDING ON WHICH AREA YOU WANT TO ANALYZE!


P = 300
subset = 500
vmin = 4.0 # min speed (cm/s) 
xmin = 0.1 # min firing rate (Hz)
# subset_neur = 24 if area == 'CA1' else 19
subset_neur = 19 
seed = 213423 if area == 'CA1' else 45342
np.random.seed(seed) 

ns = 500 # neural subsets. can set this to a much smaller value (eg 15) and still get good results in a fraction of the time.  


for g in tqdm(range(ns)): 
    rep_errs, rep_geoms, task_dims = [], [], [] 
    for p in paths:
        idx = p.split("_rawdat")[0].split("/")[-1]
        print(idx)
        out = np.load(p, allow_pickle=True).item()['dat']
        # basic analysis involving readout error
        xraw,zraw,trls,locs = out
        emps, thes, geoms, svcs = get_learning_curve(xraw,zraw,locs,dt=dt,
                                              area=area, P=P, subset=subset,
                                              vmin=vmin, 
                                              subset_neur=subset_neur,
                                            xmin=xmin)
        berrs = trls.groupby('epoch').mean().correct.values
        print('shapes = ', berrs.shape, emps.shape, thes.shape)
        rep_errs.append(np.stack([berrs, emps, thes, svcs],axis=1)) # so rep_errs will have shape n_rat x n_sesh x 4
        rep_geoms.append(geoms) # geoms has shape n_sesh x n_measures
    errs.append(rep_errs)
    all_geoms.append(rep_geoms) 
    all_tdims.append(task_dims)
    # does anyhting get messed/is there duplicate data introduced by doing this lis thing? 
    
errs = np.stack(errs) # ns x n_rat x n_sesh x n_feat
all_geoms = np.stack(all_geoms) # ns x n_rat x n_sesh x n_feat 
outpath=outdir + f'/{area}_results_nu_dt_{dt}_ns_{ns}_P_{P}.npy'
np.save(outpath, np.array({'errs' : errs, 
                           'all_geoms' : all_geoms}))


















