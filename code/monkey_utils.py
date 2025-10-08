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



####################################################################################
##################### GEOMETRY AND DATA ANALYSIS 
####################################################################################

# def theta(x): 
#     return torch.heaviside(x, values=torch.tensor([0.], device=device,dtype=dtype))

# def theory_part(Psi, Phi, Omega, P):
#     ''' 
#     theoretical calculation of generalization error
#     '''
#     T1 = np.pi/P * torch.trace(Psi@Psi.T) * torch.trace(Omega) 
#     T2 = 2 * (torch.trace(Phi.T @ Psi @ Phi) - torch.trace(Phi.T@Phi)**2/torch.trace(Omega)) 
#     T3 = np.sqrt(2) * torch.trace(Phi@Phi.T)
#     return 1/np.pi * torch.arctan(torch.sqrt(torch.trace(Omega)*(T1+T2))/T3)

# def get_geometry(Psi, Phi, Omega): 
#     ''' 
#     geometric terms 
#     '''
#     PR = torch.trace(Psi)**2 / torch.trace(Psi@Psi.T)
#     corr = torch.trace(Phi@Phi.T) / torch.trace(Omega) / torch.trace(Psi)
#     # align = torch.trace(Phi.T @ Psi @ Phi) / torch.trace(Psi) / torch.trace(Phi@Phi.T)
#     # return PR.cpu().numpy(), corr.cpu().numpy(), 1/align.cpu().numpy()  
#     Omega_inv = torch.linalg.inv(Omega)
#     H = Psi - Phi @ Omega_inv @ Phi.T 
#     fact = torch.trace(Phi.T @ Phi)**2/torch.trace(Omega) / torch.trace(Phi.T @ Phi @ Omega @ Phi.T @ Phi)
#     sna = torch.trace(Phi.T@Phi)**2 / torch.trace(Omega) / torch.trace(Phi.T @ H @ Phi) 
#     return PR.cpu().numpy(), corr.cpu().numpy(), fact.cpu().numpy(), sna.cpu().numpy()

# def emp_error(points, latents, P, n_task=500, svc_analysis=True, max_bias = 0.25, min_task=100, dtype=dtype): 
#     ''' 
#     evaluate the generalization error empirically. torch implementation for gpu. 
#     Args:
#     - points: neural responses. ('x' variables)
#     - latents: latent variables. ('z' variables) 
#     - P: number of training samples to use 
#     - n_task: number of T vectors to draw
#     '''
#     P_full, D = latents.shape 
#     teach = torch.randn(D, n_task, device=device, dtype=dtype)
#     # P x n_task labels 
#     labels = torch.sign(latents @ teach) 
#     # consider only problems with reasonably balanced labels
#     fracp,fracm = torch.mean(1*(labels==1), dtype=dtype), torch.mean(1*(labels==-1), dtype=dtype)
#     bias = torch.max(torch.stack([fracp, fracm],dim=0),dim=0) 
#     valid = torch.abs(labels.mean(0)) < max_bias
#     assert torch.sum(valid) > min_task, f"need more problems: {labels.mean(0)}"     
#     labels = labels[:, valid]
#     # split into train/test 
#     errs = []
#     idx = torch.randperm(P_full)
#     train_idx, test_idx = idx[:P], idx[P:] 
#     trainp, testp = points[train_idx], points[test_idx] 
#     trainl, testl = labels[train_idx], labels[test_idx] 
#     # n_task x N of readouts
#     W = 1/P * torch.einsum('ms, mk -> sk', trainl, trainp)
#     yhat = torch.sign(torch.einsum('sk, mk -> ms', W, testp))
#     errs = torch.mean(theta(-testl * yhat), axis=0)
#     if svc_analysis is True: 
#         # cast labels to {1,0}
#         trainl, testl = 1/2 * (1 + trainl), 1/2 * (1 + testl)
#         val = (trainl.mean(0) != 1.0) & (trainl.mean(0) != 0.)
#         if val.sum() < min_task: 
#             err_svc = np.nan
#         else: 
#             trainl, testl = trainl[:, val].cpu().numpy(), testl[:, val].cpu().numpy()
#             trainp, testp = trainp.cpu().numpy(), testp.cpu().numpy()
#             cls = MultiOutputClassifier(LinearSVC(dual='auto'), n_jobs=-1)
#             cls.fit(trainp, trainl)
#             pred = cls.predict(testp) 
#             err_svc = np.mean(pred != testl)
#             # err_svc = cls.score(testp.cpu().numpy(), testl.cpu().numpy())
#             print(err_svc) 
#         return errs.mean(), err_svc, errs.std(), errs
#     else:
#         return errs.mean(), errs.std(), errs

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
