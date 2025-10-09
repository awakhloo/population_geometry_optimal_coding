#!/usr/bin/env python

#SBATCH --time=1-00:00:00
#SBATCH --partition ccn
#SBATCH --nodes=1
#SBATCH -C icelake

from pynwb import NWBHDF5IO, NWBFile, TimeSeries
from pynwb.behavior import Position, SpatialSeries
from pynwb.file import Subject
import numpy as np 
from tqdm import tqdm
import pandas as pd
from scipy.ndimage import gaussian_filter
from scipy import stats
from scipy.special import erfc
from scipy.interpolate import UnivariateSpline
import matplotlib.pyplot as plt
from glob import glob
from scipy.integrate import dblquad
from sklearn.covariance import EmpiricalCovariance, MinCovDet, GraphicalLassoCV, GraphicalLasso
import sys
import os 
sys.path.append(os.getcwd()) 

import rat_utils as rat 

np.random.seed(7496)

outdir = '/mnt/home/awakhloo/ceph/reprod_population_geom_opt_coding/results' # path to results 
dat_dir = '/mnt/home/awakhloo/ceph/abstraction/data' # path to W-track rat data in nwb format
paths = ['/000978/sub-JDS-SingleDay-KL8/sub-JDS-SingleDay-KL8_behavior+ecephys.nwb',
         '/000978/sub-JDS-SingleDay-ER1/sub-JDS-SingleDay-ER1_behavior+ecephys.nwb',
         '/000978/sub-JDS-SingleDay-JS34/sub-JDS-SingleDay-JS34_behavior+ecephys.nwb',
         '/000978/sub-JDS-SingleDay-JS14/sub-JDS-SingleDay-JS14_behavior+ecephys.nwb',
         '/000978/sub-JDS-SingleDay-JS17/sub-JDS-SingleDay-JS17_behavior+ecephys.nwb',
         '/000978/sub-JDS-SingleDay-JS21/sub-JDS-SingleDay-JS21_behavior+ecephys.nwb',
         '/000978/sub-JDS-SingleDay-JS15/sub-JDS-SingleDay-JS15_behavior+ecephys.nwb']
paths = [dat_dir + p for p in paths]

# dt = 1/4 # 250ms bins 
dt = 1/2
outpath = outdir + f'/processed_rat_dat/{str(dt)}'
os.makedirs(outpath, exist_ok=True)


for path in paths: 
    # make the directory 
    idx = os.path.dirname(path)[-4:].strip('-')
    # load data and bin
    io = NWBHDF5IO(path, "r")
    f = io.read()
    out = rat.get_trl_dat(f, dt=dt) 
    np.save(outpath + f'/{idx}_rawdat.npy', np.array({'dat' : out}))
    

# one of the rats has its data spread across two files; treat this one separately 
paths = ['ceph/abstraction/data/000978/sub-JDS-SingleDay-ZT2/sub-JDS-SingleDay-ZT2_obj-u40err_behavior+ecephys.nwb',
         'ceph/abstraction/data/000978/sub-JDS-SingleDay-ZT2/sub-JDS-SingleDay-ZT2_obj-1dss6zi_behavior+ecephys.nwb']
paths = ['/mnt/home/awakhloo/' + p for p in paths]
idx = os.path.dirname(paths[0])[-4:].strip('-')
all_outs = [] 
last_epoch = 0
for path in paths: 
    io = NWBHDF5IO(path, "r")
    f = io.read()
    out = rat.get_trl_dat(f, dt=dt)
    # align epochs correctly. note that we add 2 since the epochs typically correspond to odd integers, 1, 3, 5, etc 
    out[2]['epoch'] = out[2]['epoch'] + last_epoch + 2 
    last_epoch = out[2]['epoch'].max() 
    all_outs.append(out) 
both_rasters = sum([all_outs[i][0] for i in range(len(paths))], []) 
both_behvs = sum([all_outs[i][1] for i in range(len(paths))], []) 
both_trls = pd.concat([all_outs[i][2] for i in range(len(paths))], axis=0)
both_locs = sum([all_outs[i][3] for i in range(len(paths))], []) 
Zout = (both_rasters, both_behvs, both_trls, both_locs) 
np.save(outpath + f'/{idx}_rawdat.npy', np.array({'dat' : Zout}))
    















