import numpy as np 
import torch
from matplotlib import image
from tqdm import tqdm 
from brainscore_vision import load_benchmark
import gc
device = 'cuda' if torch.cuda.is_available() else 'cpu'
dtype=torch.float64
# from sc
from sklearn.multioutput import MultiOutputClassifier
from sklearn.svm import LinearSVC
from utils import theta_torch, theory_part, get_geometry, emp_error_torch


def get_learning_curves(x,z,Ps,nrep=15,dis=False):
    '''
    get theoretical and empirical generalization errors. 
    '''
    # x,z = torch.tensor(x,device=device), torch.tensor(z,device=device) 
    Pall = x.shape[0] 
    Psi = x.T @ x / Pall 
    Phi = x.T @ z / Pall 
    Omega = z.T @ z / Pall
    Psi = Psi.cpu().numpy()
    Phi = Phi.cpu().numpy()
    Omega = Omega.cpu().numpy()
    geom = get_geometry(Psi=Psi, Phi=Phi, Omega=Omega) 
    
    errs_emp = torch.zeros((nrep, len(Ps)), device=device)
    errs_the = torch.zeros(len(Ps), device=device)
    errs_svc = torch.zeros((nrep, len(Ps)), device=device) 
    seedos = np.random.randint(low=0,high=60_000, size=(nrep,))
    for n in tqdm(range(nrep), disable=dis): 
        torch.manual_seed(seedos[n])
        for i, P in enumerate(Ps): 
            m,s, *_ = emp_error_torch(x,z,P)
            errs_emp[n,i] = m 
            errs_svc[n,i] = s
            errs_the[i] = theory_part(Psi, Phi, Omega, P)
    return errs_emp.cpu().numpy(), errs_svc.cpu().numpy(), errs_the.cpu().numpy(), geom 


###############################################################

##################### DATA PROCESSING 
###############################################################

def filter_data(assm, view_lat, cl,dtype=dtype, stim=None, random_proj=False, proj_dim=88): 
    ''' 
    normalize and filter by category name
    Args:
    - assm: assembly object
    - view_lat: latent variables to slice
    - cl: class label 
    ''' 
    # z-score latents
    pz = np.stack([assm.coords[l].values for l in view_lat],axis=1)
    pz = (pz-pz.mean(0)) / pz.std(axis=0,keepdims=True)
    # mean center neural data (or pixels if stim is passed) 
    neur = assm.values
    if stim is not None: 
        neur = stim  
    # do a random projection to 88 dimensions
    if random_proj == True: 
        M = np.random.randn(neur.shape[1], proj_dim) 
        neur = neur @ M  
        print(neur.shape)
    neur = neur - neur.mean(0)
    # either filter for class or dont
    if cl is None: 
        mask = assm.coords['variation'] == 3 
    else : 
        mask = (assm.coords['variation'] == 3 ) & (assm.coords['category_name'].values.astype(str) == cl)
    return torch.tensor(pz[mask],device=device,dtype=dtype), torch.tensor(neur[mask],device=device,dtype=dtype)

def rgb2gray(rgb):
    '''
    convert from rgb to greyscale
    '''
    return np.dot(rgb[...,:3], [0.2989, 0.5870, 0.1140])    

def load_all_stimuli(view_lat, dtype=dtype): 
    '''
    load all images
    '''
    benchmark = load_benchmark('MajajHong2015public.V4-pls')    
    assm = benchmark._assembly
    stimulus_set = assm.attrs['stimulus_set']
    stim_ids = assm['stimulus_id'].values
    imgs = np.zeros((len(stimulus_set), 256,256))
    for i, stim_id in tqdm(enumerate(stim_ids)): 
        local_path = stimulus_set.get_stimulus(stim_id)
        img = image.imread(local_path) 
        imgs[i] = rgb2gray(img)
    return imgs.reshape(imgs.shape[0], -1)

def random_proj(X, dim=5_000): 
    ''' 
    gaussian random projection
    '''
    p, d = X.shape 
    X = torch.tensor(X, dtype=dtype, device=device) 
    M = torch.randn(d,dim,dtype=dtype,device=device)/np.sqrt(dim) 
    return (X @ M).cpu().numpy()


############### HELPER FUNCTIONS 


def avg_all_cls(assm,view_lat,Ps,dtype=dtype,stim=None,random_proj=False): 
    ''' 
    carry out the analysis, subsetting by individual superclass categories
    '''
    all_cls = np.unique(assm.coords['category_name'].values).astype(str)
    emps, svcs, thes, geoms = [], [], [] ,[] 
    for cl in tqdm(all_cls): 
        gc.collect() 
        z, x = filter_data(assm, view_lat, cl, stim=stim, dtype=dtype,random_proj=random_proj) 
        emp, svc, the, geom = get_learning_curves(x,z,Ps, dis=True) 
        emps.append(emp), svcs.append(svc), thes.append(the), geoms.append(geom) 
    return np.stack(emps), np.stack(svcs), np.stack(thes), np.stack(geoms) 

def pool_all_cls(assm,view_lat,Ps,dtype=dtype,stim=None): 
    '''
    redo analysis, pooling data from all classes together 
    '''
    z, x = filter_data(assm, view_lat, cl=None, stim=stim, dtype=dtype) 
    emp, svc, the, geom = get_learning_curves(x,z,Ps, dis=False) 
    return emp, svc, the, geom
