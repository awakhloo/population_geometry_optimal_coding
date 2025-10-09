import numpy as np 
from tqdm import tqdm
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.multioutput import MultiOutputClassifier
from sklearn.svm import LinearSVC

# def theta(x): 
#     return (x>0)*1

def relu(x): 
    return x * (x>0)

# def theory_part(Psi, Phi, Omega, P):
#     T1 = np.pi/P * np.trace(Psi@Psi.T) * np.trace(Omega) 
#     T2 = 2 * (np.trace(Phi.T @ Psi @ Phi) - np.trace(Phi.T@Phi)**2/np.trace(Omega)) 
#     T3 = np.sqrt(2) * np.trace(Phi@Phi.T)
#     return 1/np.pi * np.arctan(np.sqrt(np.trace(Omega)*(T1+T2))/T3)

# def emp_error(points, latents, P, n_task=500, svc_analysis=False, max_bias = 0.25, min_task=100): 
#     P_full, D = latents.shape 
#     teach = np.random.randn(D, n_task)
#     # P x n_task labels 
#     labels = np.sign(latents @ teach) 
#     # consider only problems with reasonably balanced labels
#     fracp,fracm = np.mean(labels==1,), np.mean(labels==-1)
#     bias = np.max(np.stack([fracp, fracm],axis=0),axis=0) 
#     valid = np.abs(labels.mean(0)) < max_bias
#     assert np.sum(valid) > min_task, f"need more problems: {labels.mean(0)}" 
#     labels = labels[:, valid]
#     # subsample 
#     errs = []
#     idx = np.arange(P_full)
#     train_idx = np.random.choice(idx, replace=False, size=(P,))
#     # sample train and test set 
#     test_idx = idx[~np.isin(idx, train_idx)]
#     trainp, testp = points[train_idx], points[test_idx] 
#     trainl, testl = labels[train_idx], labels[test_idx] 
#     # n_task x N of readouts
#     W = 1/P * np.einsum('ms, mk -> sk', trainl, trainp)
#     yhat = np.sign(np.einsum('sk, mk -> ms', W, testp))
#     errs = np.mean(theta(-testl * yhat), axis=0)
#     if svc_analysis is True: 
#         # cast labels to {1,0}
#         trainl, testl = 1/2 * (1 + trainl), 1/2 * (1 + testl)
#         cls = MultiOutputClassifier(LinearSVC(dual='auto',max_iter=10_000), n_jobs=-1)
#         cls.fit(trainp, trainl)
#         pred = cls.predict(testp)
#         err_svc = np.mean(pred != testl)
#         print("ERR SVC = ", err_svc)
#         return errs.mean(), err_svc, errs.std(), errs, 
#     else:
#         return errs.mean(), errs.std(), errs

# def get_geometry(Psi, Phi, Omega, p):
#     Om_inv = np.linalg.inv(Omega) 
#     dim = np.trace(Psi)**2 / np.sum(Psi**2) 
#     ssf = np.trace(Phi@Phi.T)**2 / np.trace(Omega) / np.trace(Phi.T @ Phi @ Om_inv @ Phi.T @ Phi)
#     snf = np.trace(Phi@Phi.T)**2 / np.trace(Omega) / np.trace(Phi.T @(Psi - Phi@ Om_inv @ Phi.T) @ Phi)
#     c = np.trace(Phi@Phi.T) / np.trace(Psi) / np.trace(Omega) 
#     eg = 1/np.pi * np.arctan(np.sqrt(np.pi / (2*p*c**2*dim) + 1/snf + 1/ssf -1))
    
#     task_cov = Phi @ np.linalg.inv(Omega) @ Phi.T 
#     task_dim = np.trace(task_cov)**2 / np.sum(task_cov**2)
#     task_dim = task_dim / Omega.shape[0]
#     zdim = np.trace(Omega)**2/np.sum(Omega**2)/Omega.shape[0]
    
#     Phi_hat = Phi / np.linalg.norm(Phi, axis=0)
#     D = Omega.shape[0]
#     fact_adhoc = (np.sum((Phi_hat.T @ Phi_hat) ** 2) - D) / (D * (D - 1))
    
#     return dim, c, snf, ssf, task_dim, zdim, fact_adhoc, eg 




##############################################################################

################## DATA PROCESSING ##########################################

#########################################################################

def gker(x,sigma):
    return np.exp(-1/(2*sigma**2) * x**2) / np.sqrt(2*np.pi*sigma**2) 

def nadaraya(x, y, x_eval, h, ker=gker): 
    ''' 
    one dimensional nadaraya smoothing 
    '''
    assert len(x.shape) == len(y.shape) 
    assert len(x.shape) == 1, "need one d data!" 
    dists = x_eval[:, None] - x[None, :] 
    ker_evals = ker(dists, h) 
    return (ker_evals * y[None, :]).mean(1) / ker_evals.mean(1)     

def process_single_epoch(epoch, this_epoch, units, times, bdat, dt, s=None, sigma_v=0.7, sigma_l=0.05):
    ''' 
    process a single behavioral epoch
    Args: 
    - epoch: index [i] of rthe epoch. Used for error msgs 
    - this_epoch: subset of trials dataframe containing start and stop times for this epoch
    - a dataframe containing the spike times of each unit where each row is a putative single unit
    - the time at which each frame was taken - shape is (nframes). This is used to align behavior to neural responses
    - behavioral time series - shape is (nframes x nfeatures)  
    - bin size
    '''
    # get start and stop times 
    start, stop = this_epoch.start_time.min(), this_epoch.stop_time.max() 
    assert np.allclose(this_epoch.start_time[1:], this_epoch.stop_time[:-1]), f"found time where the animal wasn't doing a trial during  epoch {epoch}. Trial deltas = {this_epoch.start_time[1:]-this_epoch.stop_time[:-1]}" # double check that between start_time and stop_time there were only actual trials.
    print(f'epoch {epoch} lasted ', (stop - start)/60, ' minutes')
    
    # bin the neural data  
    # T = np.linspace(start, stop, int((stop-start)/dt))
    T = start + np.arange(int((stop-start)/dt)) * dt
    assert T.max() <= stop 
    raster = np.zeros((units.shape[0], len(T)-1))
    for u in range(units.shape[0]):
        spks = units.iloc[u].spike_times
        raster[u] = np.histogram(spks, bins=T)[0]
        
    # interpolate the behavioral data to align with the midpoint of these temporal bins
    mask = (times >= start) & (times <= stop)
    this_trl_behv = bdat[mask]   
    trl_times = times[mask]
    x,y,v = this_trl_behv[:,0], this_trl_behv[:,1], this_trl_behv[:,2]
    # estimate the value of each behavioral variable at the center of each time bin 
    btime = T[:-1] + dt/2
    nx, ny, nv = nadaraya(trl_times, x, btime, h=sigma_l), nadaraya(trl_times, y, btime, h=sigma_l), nadaraya(trl_times, v, btime, h=sigma_l)

    # calculate velocity using a more aggressive smoothing protocol 
    dx, dy = (x[1:] - x[:-1]) / (trl_times[1:] - trl_times[:-1]), (y[1:] - y[:-1]) / (trl_times[1:] - trl_times[:-1])
    vel_time = (trl_times[1:] + trl_times[:-1])/2 # take the midpoint 
    ndx, ndy = nadaraya(vel_time, dx, btime, h=sigma_v), nadaraya(vel_time, dy, btime, h=sigma_v)
    
    # add a 1/0 representation of accuracy and trajectory type at each timepoint
    accs = np.zeros(len(btime)) + np.nan
    traj = accs.copy() 
    for trl in range(this_epoch.shape[0]): 
        s,f = this_epoch.iloc[trl][['start_time', 'stop_time']].values
        mask = (btime>=s)&(btime<f)
        accs[mask] = this_epoch.iloc[trl].correct.item() 
        traj[mask] = this_epoch.iloc[trl].trajectory_type.item() 
    # accs = accs[:-1] # drop the last time step since we're interested in the midpoint of the bins 
    assert np.all(~np.isnan(accs)), f"nan accuracy values..., {f}, {btime.max()}, {T.max()}" 
    z = np.stack([nx,ny,nv,ndx,ndy,accs,traj], axis=1) 
    return raster.T, z, this_trl_behv, trl_times, btime

def get_trl_dat(f,dt,lower=None,upper=None): 
    '''
    get behavioral sequences and spike times during every trial
    ''' 
    # get trial and epoch data 
    trls = f.trials.to_dataframe() 
    intvs=f.intervals['epoch intervals'].to_dataframe()
    trls['epoch'] = 0. 
    for i in range(1,intvs.shape[0] , 1): 
        targ_int = intvs.iloc[i] 
        mask = (trls.start_time >= targ_int.start_time) & (trls.stop_time <= targ_int.stop_time)
        trls.loc[mask, 'epoch'] = i
    # gather spike times for all units and behavior 
    units = f.units[:]
    behv = f.processing['behavior']['Position']['SpatialSeries']
    times = behv.timestamps[:]  
    bdat = behv.data[:]
    all_rasters = [] 
    all_behavior = [] 
    all_locs = [] 
    # use only the epochs that correspond to trials 
    for epoch in trls.epoch.unique(): 
        this_epoch = trls[trls.epoch == epoch] 
        raster, behavior, *_ = process_single_epoch(epoch, this_epoch,units,times,bdat,dt=dt)
        all_rasters.append(raster)
        all_behavior.append(behavior)
        all_locs.append(get_unit_locations(f)) # for each session, create a copy of the unit locations. This is basically a workaround to deal with rat ZT2, which has two different sets of units across diff sessions
    return all_rasters, all_behavior, trls, all_locs


def bin_dat(z,x,bin_scale,vmin=4,min_trls=8) : 
    ''' 
    Do spatial binning on the data. The idea is to average neural activity
    over distinct visits to the same spatial bin. 
    Args:
    - z: positions for each "trial." Code assumes that the first two columns contain the (x,y) coordinates of the animal
    - bin_scale: inverse size of the spatial bins (cm^2) 
    - vmin: minimum velocity to consider 
    - min_trls: minimum amount of trials a bin must have to be considered
    '''
    assert z.shape[1] == 3, "should only pass x, y position and speed as latents" 
    all_var = np.concatenate([z[:,:2],x],axis=1)
    df= pd.DataFrame(all_var, columns=['xpos', 'ypos'] + [f'neur_{i}' for i in range(x.shape[1])])
    # consider only locomotion periods 
    mask = z[:, 2] > vmin 
    df = df[mask] 
    df['xpos'] = (df.xpos*bin_scale).round(0)/bin_scale 
    df['ypos'] = (df.ypos*bin_scale).round(0)/bin_scale 
    rounded_locs=df.iloc[:,:2].values
    # keep only trials that have 
    unique_locs, counts = np.unique(rounded_locs, axis=0, return_counts=True)
    print("frac kept = ", np.sum(counts >= min_trls) / counts.shape[0])
    keep = unique_locs[counts >= min_trls] # locations to keep. shape is now (n_unique x 2)
    mask = (rounded_locs[None, :, :] == keep[:, None, :]).all(2).any(0) # compare z and unique values. make sure both entries match (.all(2)), for any of the unique values (.any(0)) 
    # trial average over the remaining locations 
    df = df[mask]
    df = df.groupby(['xpos', 'ypos']).mean().reset_index() 
    z,x = df.iloc[:, :2].values, df.iloc[:,2:].values
    return z,x
    

def process_dat(z,x,locs,dt,area='CA1',vmin=4,xmin=0.1,xmax=np.inf):  
    ''' 
    subset by area, drop dead units, z-score latents, and consider only locomotion periods.
    Args: 
    - z: an array of latents. The x and y position of the animal should be in the first two columns
    - x: neural activity represented as binned spikes 
    - locs: a vector of strings describing whether each unit is in CA1 or PFC 
    - dt: size of temporal binning of the spikes
    - area: what area to take neurons from. If none, pool data from PFC+CA1
    - vmin: minimal velocity (in cm/s) of the animal to conisder. 
    '''
    #  only consider locomotion periods
    tinit = z.shape[0] 
    # mask = z[:, 2] > vmin 
    mask = np.linalg.norm(z[:, 3:5], axis=1) > vmin
    z,x = z[mask], x[mask]
    tpost = z.shape[0] 
    print('dropping ', tinit - tpost, ' out of ', tinit, ' trials due to locomotion constraint')
    
    #### take only one area
    assert locs.shape[0] == x.shape[1], f"location indices does not align with neural indices. Got {locs.shape[0]} loc indices and {x.shape[1]} neural indices"
    if area is not None: 
        valid = locs == area 
        x = x[:, valid]
        
    # drop dead units. note that we divide by bin size so that xmin and xmax are the cutoffs in Hz. 
    x = x[:, (x.mean(0)/dt > xmin) & (x.mean(0)/dt < xmax)]
    print("MINIMAL FIRING RATE = ", x.mean(0).min()/dt, x.std(0).min()/dt)
    print("NUM FEASIBLE CELLS  = ", x.shape[1])
    print('neur rate mean std = ', x.mean()/dt, x.std()/dt)
    eps = 0.01 # eps /dt Hz regularizer 
    x = x-x.mean(0,keepdims=True) 
    x = x/(eps+x.std(0,keepdims=True))
    
    # z-score latents
    z = (z - z.mean(0,keepdims=True))/ z.std(0,keepdims=True) 
    return x,z 

def get_unit_locations(f): 
    n_units = f.units.to_dataframe().shape[0]
    locs = [] 
    elec=f.units.to_dataframe().electrodes
    for i in range(n_units):
        nam=elec[i].location.item()
        locs.append(nam)
    return np.array(locs)


# def svc_analysis(z,x,n_task=1000,min_task=300,cv=2): 
#     ''' 
#     calculate Eg using a single cross-val split 
#     ''' 
#     p,d = z.shape
#     r = np.random.randn(d,n_task) 
#     lab = ((1+np.sign(z @ r))/2).astype(int)
#     # throw out extremely biased tasks to avoid SVM convergence problems 
#     lab = lab[:, np.abs(lab.mean(0)-1/2) < 0.2] 
#     cls=LinearSVC()
#     return np.mean(1-cross_val_score(cls, x, lab[:,0], cv=cv,error_score='raise'))






############
#### GLM MODEL FITTING 
#############

def get_all_lat(paths, area, vmin, n_sesh): 
    ''' 
    return nested list with rat x session x latents as well as a flattened list with rat x latents 
    '''
    all_latents = []
    all_x = [] 
    for i, path in tqdm(enumerate(paths)):
        this_rat_lat, this_rat_x = [], []
        arr = np.load(path,allow_pickle=True).item()['dat']
        xr,zr,trl,locs = arr 
        for j, sesh in enumerate(range(n_sesh)): 
            x,z,loc = xr[sesh], zr[sesh], locs[sesh]
            # velocity threshold 
            mask = z[:,2] > vmin
            x,z = x[mask], z[mask] 
            # subset by area
            assert loc.shape[0] == x.shape[1], "Locations and neurons misaligned"
            x = x[:, loc == area]
            # mean center positions 
            z[:,:2] = z[:,:2] - z[:,:2].mean(0)
            this_rat_lat.append(z) 
            this_rat_x.append(x) 
        all_latents.append(this_rat_lat) 
        all_x.append(this_rat_x)
    return all_latents, all_x, [np.concatenate(zh) for zh in all_latents]

def gauss_basis(z,m,s): 
    return 1/np.sqrt(2*np.pi*s) * np.exp(-1/(2*s) *(z-m)**2)

def iso_gauss(z,m,s): 
    '''isotropic gaussian. assumes z has the feature dimension on the right''' 
    d = z.shape[-1] 
    return 1/np.sqrt(2*np.pi*s)**d * np.exp(-1/(2*s) * np.linalg.norm(z-m,axis=-1)**2)

def relu(x): 
    return (x>0)*x

def get_covs(z,basis_fns_posn,basis_fns_vx,basis_fns_vy,split_traj):
    posn, vx, vy = z[:,:2], z[:,3], z[:,4]
    posn_covs = np.stack([b(posn) for b in basis_fns_posn],axis=1)
    vx_covs = np.stack([b(vx) for b in basis_fns_vx],axis=1)
    vy_covs = np.stack([b(vy) for b in basis_fns_vy],axis=1)
    if split_traj == True: 
        # fit separate place maps for each trajectory type 
        traj_type = z[:,6] 
        posn_covs_0, posn_covs_1 = posn_covs.copy(), posn_covs.copy() 
        posn_covs_0[traj_type==1] = 0 
        posn_covs_1[traj_type==0] = 0 
        covs = np.concatenate([posn_covs_0, posn_covs_1,vx_covs,vy_covs],axis=1)
    else:
        covs = np.concatenate([posn_covs, vx_covs, vy_covs],axis=1)
    return covs 

def bin_positions(posn, bin_size, min_count): 
    ''' 2d binning for space ''' 
    binned = (posn / bin_size).round(0) * bin_size 
    bins, counts = np.unique(binned,axis=0,return_counts=True)
    bins = bins[counts>min_count]
    return bins 

def bin_velocities(vels, bin_size_vel, min_count): 
    ''' 1 dimensional binning for the velocity''' 
    binned = (vels / bin_size_vel).round(0) * bin_size_vel
    bins,counts = np.unique(binned,return_counts=True) 
    bins = bins[counts>min_count]
    return bins 

def plot_spatial(basis_fns, bins, posn, K=150,L=70):
    # plot some basis functions
    x,y = np.linspace(-L,L,K), np.linspace(-L,L,K)
    x,y = np.repeat(x, K), np.tile(y, K)
    posn_hat = np.stack([x,y],axis=1)
    inds = np.arange(1,len(basis_fns),4)
    cs = [] 
    for ind in inds: 
        cs.append(basis_fns[ind](posn_hat))
        
    cfull = np.stack(cs).sum(0)
    plt.scatter(x,y,c=cfull,cmap='RdPu',alpha=0.5)
    
    plt.scatter(posn[:,0], posn[:,1], alpha=0.05)
    plt.scatter(bins[:,0],bins[:,1],alpha=.8,color='purple')
    
    print('n-bin space = ', bins.shape[0])
    plt.xlim(-L,L)
    plt.ylim(-L,L)
    plt.title("spatial covariates")
    plt.show()

def plot_velocity(basis_fns_vx, basis_fns_vy, bins_vx, bins_vy, vels,L=60, fn=16): 
    B = 50 
    fig, ax = plt.subplots(1,2,figsize=(15,5))
    ax[0].hist(vels[:,0],bins=B,density=True) 
    ax[1].hist(vels[:,1],bins=B,density=True)
    for i, a in enumerate(ax.flat):
        a.set_title(['x', 'y'][i] + '-velocity',fontsize=fn)
    X = np.linspace(-L,L,100) 
    for i,bases in enumerate([basis_fns_vx, basis_fns_vy]): 
        for b in bases: 
            ax[i].plot(X, b(X), ls='--', color='grey')
        ax[i].set_xlim([-L,L])



def get_pr(xh, eps = 0.01): 
    if np.all(xh.std(0) == 0.): 
        print("got all 0")
        return 0. 
    xh = (xh - xh.mean(0))/(xh.std(0)+eps) 
    Psi = xh.T @ xh / xh.shape[0] 
    pr = np.trace(Psi)**2 / np.sum(Psi**2) 
    return pr

def format_glm_results(all_res, subset_neur, frac_val=0.9, n_sesh=8, n_rat=8, n_subset=100, silent=False):
    ''' 
    calculate task-dimensionality and average cross-validated error, substituting in the null model where appropriate. 
    subset neurons as appropriate
    '''
    dims = np.zeros((n_rat, n_sesh,2))
    alphas = np.zeros((n_rat, n_sesh))
    test_scores = np.zeros((n_rat, n_sesh))
    test_devs = np.zeros((n_rat, n_sesh))
    for i in range(8):
        if silent==False: 
            print(f"rat {i}")
        for s in range(n_sesh): 
            res = all_res[i][s] 
            alphas[i,s] = res['alphas'].max() 
            x, xh = res['x'], res['xhat']
            cv = res['glm_params'][-1]
            # (note that test scores obtains the nan-mean cross-val score for each unit (axis=0) across different alphas (axis=1) 
            best_inds = np.argmax(res['test_scores'],axis=1)
            x_inds = np.arange(res['test_scores'].shape[0])
            best_scores = res['test_scores'][x_inds, best_inds]
            num_val = res['num_valid'][x_inds, best_inds]
            # get the average test score across all units as well as the SEM averaged across all units. 
            test_scores[i,s] = best_scores.mean() # avg best score across units 
            test_devs[i,s] = np.mean(res['test_devs'][x_inds, best_inds]/ np.sqrt(res['num_valid'][x_inds, best_inds]))    

            # set any units which: (1) had a best score less than the intercept model, or (2), did not have sufficiently many valid cv splits to 0. 
            zero_mask = (best_scores <= 0) | (num_val < frac_val * cv) | np.isnan(best_scores)
            xh[:, zero_mask] = 0.
            
            # calculate subsetted PR
            PRs = np.zeros((n_subset,2))
            for k in range(n_subset): 
                sub_x = np.random.choice(np.arange(x.shape[1]), size=(subset_neur,), replace=False)
                sub_xh = np.random.choice(np.arange(xh.shape[1]), size=(subset_neur,), replace=False)
                pr_x, pr_xh = get_pr(x[:, sub_x]), get_pr(xh[:, sub_xh]) 
                PRs[k] = np.array([pr_x,pr_xh])
            dims[i,s] = PRs.mean(0)
    return dims, alphas, test_scores, test_devs, res['glm_params']


def find_best_params(all_paths, subset_neur): 
    ''' 
    loop through glm model fitting results and get the nan-meaned average test error for each model. 
    '''
    all_params, all_scores = [], [] 
    for R in tqdm(all_paths): 
        arr = np.load(R,allow_pickle=True).item()['results']
        dims,alphas,test_scores,test_devs,params = format_glm_results(arr,subset_neur, silent=True)
        all_params.append(params); all_scores.append(np.nanmean(test_scores))
    return all_params, all_scores
          

