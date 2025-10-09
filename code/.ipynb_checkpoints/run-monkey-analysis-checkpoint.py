#!/usr/bin/env python
#SBATCH --time=1-00:00:00
#SBATCH --partition ccn
#SBATCH --nodes=1
#SBATCH -C icelake

import numpy as np 
import torch
import os
from brainscore_vision import load_benchmark
device = 'cuda' if torch.cuda.is_available() else 'cpu'
dtype=torch.float64

import sys 
import os 
sys.path.append(os.getcwd())
import monkey_utils as monk 

np.random.seed(57392)

outdir= '/mnt/home/awakhloo/ceph/reprod_population_geom_opt_coding/results'
outpath = outdir+'/majaj_res.npy'

view_lat = ['rxy', 'ryz', 'rxz', 's', 'tz', 'ty']

stim = monk.load_all_stimuli(view_lat) 
V4 = monk.load_benchmark('MajajHong2015public.V4-pls')._assembly
IT = monk.load_benchmark('MajajHong2015public.IT-pls')._assembly

dimstim=500 
stim_proj = monk.random_proj(stim, dim=dimstim) # change back to 5k 

Ps = np.logspace(1, np.log10(200),10, dtype=int) 
# ALL CLASSES
emppixmed, svcpixmed, thepixmed, geompixmed = monk.avg_all_cls(V4,view_lat,Ps,dtype=dtype,stim=stim_proj)
empv4med, svcv4med, thev4med, geomv4med = monk.avg_all_cls(V4,view_lat,Ps,dtype=dtype,stim=None)
empITmed, svcITmed, theITmed, geomITmed = monk.avg_all_cls(IT,view_lat,Ps,dtype=dtype,stim=None)

# global pooling
Psg = np.logspace(1, 3, 15,dtype=int)
pemppixmed, psvcpixmed, pthepixmed, pgeompixmed = monk.pool_all_cls(V4,view_lat,Psg,dtype=dtype,stim=stim_proj)
pempv4med, psvcv4med, pthev4med, pgeomv4med = monk.pool_all_cls(V4,view_lat,Psg,dtype=dtype,stim=None)
pempITmed, psvcITmed, ptheITmed, pgeomITmed = monk.pool_all_cls(IT,view_lat,Psg,dtype=dtype,stim=None)


### 88 proj 
# ALL CLASSES; RANDOM PROJECTION DOWN TO 88 UNITS
stim_proj_88 = monk.random_proj(stim, dim=88)
emppixmedproj, svcpixmedproj, thepixmedproj, geompixmedproj = monk.avg_all_cls(V4,view_lat,Ps,dtype=dtype,stim=stim_proj_88)
empITmedproj, svcITmedproj, theITmedproj, geomITmedproj = monk.avg_all_cls(IT,view_lat,Ps,dtype=dtype,stim=None,random_proj=True)


results = {'all_classes' : [[emppixmed, svcpixmed, thepixmed, geompixmed],
                            [empv4med, svcv4med, thev4med, geomv4med],
                            [empITmed, svcITmed, theITmed, geomITmed]],
           'global_pooling' : [[pemppixmed, psvcpixmed, pthepixmed, pgeompixmed],
                               [pempv4med, psvcv4med, pthev4med, pgeomv4med],
                               [pempITmed, psvcITmed, ptheITmed, pgeomITmed]],
           'proj_88' : [[emppixmedproj, svcpixmedproj, thepixmedproj, geompixmedproj],
                        [empITmedproj, svcITmedproj, theITmedproj, geomITmedproj]]}


np.save(outpath, np.array(results))







