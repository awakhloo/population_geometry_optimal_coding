import numpy as np 
import torch 
from sklearn.multioutput import MultiOutputClassifier
from sklearn.svm import LinearSVC


device = 'cuda' if torch.cuda.is_available() else 'cpu'
dtype=torch.float64


##### NUMPY IMPLEMENTATIONS

def theta(x): 
    return (x>0)*1

def theory_part(Psi, Phi, Omega, P):
    T1 = np.pi/P * np.trace(Psi@Psi.T) * np.trace(Omega) 
    T2 = 2 * (np.trace(Phi.T @ Psi @ Phi) - np.trace(Phi.T@Phi)**2/np.trace(Omega)) 
    T3 = np.sqrt(2) * np.trace(Phi@Phi.T)
    return 1/np.pi * np.arctan(np.sqrt(np.trace(Omega)*(T1+T2))/T3)


def emp_error(points, latents, P, n_task=500, svc_analysis=False, max_bias = 0.25, min_task=100): 
    P_full, D = latents.shape 
    teach = np.random.randn(D, n_task)
    # P x n_task labels 
    labels = np.sign(latents @ teach) 
    # consider only problems with reasonably balanced labels
    fracp,fracm = np.mean(labels==1,), np.mean(labels==-1)
    bias = np.max(np.stack([fracp, fracm],axis=0),axis=0) 
    valid = np.abs(labels.mean(0)) < max_bias
    assert np.sum(valid) > min_task, f"need more problems: {labels.mean(0)}" 
    labels = labels[:, valid]
    # subsample 
    errs = []
    idx = np.arange(P_full)
    train_idx = np.random.choice(idx, replace=False, size=(P,))
    # sample train and test set 
    test_idx = idx[~np.isin(idx, train_idx)]
    trainp, testp = points[train_idx], points[test_idx] 
    trainl, testl = labels[train_idx], labels[test_idx] 
    # n_task x N of readouts
    W = 1/P * np.einsum('ms, mk -> sk', trainl, trainp)
    yhat = np.sign(np.einsum('sk, mk -> ms', W, testp))
    errs = np.mean(theta(-testl * yhat), axis=0)
    if svc_analysis is True: 
        # cast labels to {1,0}
        trainl, testl = 1/2 * (1 + trainl), 1/2 * (1 + testl)
        cls = MultiOutputClassifier(LinearSVC(dual='auto',max_iter=10_000), n_jobs=-1)
        cls.fit(trainp, trainl)
        pred = cls.predict(testp)
        err_svc = np.mean(pred != testl)
        return errs.mean(), err_svc, errs.std(), errs, 
    else:
        return errs.mean(), errs.std(), errs

def get_geometry(Psi=None, Phi=None, Omega=None, p=1):
    Om_inv = np.linalg.inv(Omega) 
    dim = np.trace(Psi)**2 / np.sum(Psi**2) 
    ssf = np.trace(Phi@Phi.T)**2 / np.trace(Omega) / np.trace(Phi.T @ Phi @ Om_inv @ Phi.T @ Phi)
    snf = np.trace(Phi@Phi.T)**2 / np.trace(Omega) / np.trace(Phi.T @(Psi - Phi@ Om_inv @ Phi.T) @ Phi)
    c = np.trace(Phi@Phi.T) / np.trace(Psi) / np.trace(Omega) 
    eg = 1/np.pi * np.arctan(np.sqrt(np.pi / (2*p*c**2*dim) + 1/snf + 1/ssf -1))
    
    task_cov = Phi @ np.linalg.inv(Omega) @ Phi.T 
    task_dim = np.trace(task_cov)**2 / np.sum(task_cov**2)
    task_dim = task_dim / Omega.shape[0]
    zdim = np.trace(Omega)**2/np.sum(Omega**2)/Omega.shape[0]

    # CORRECTED ad-hoc factorization for when Omega is not identity
    # this does a weighted average of squared cosines weighted by
    # the signal strength of each latent, which ends up mattering a lot
    # for deeplabcut and the rat data
    coding_dir = Phi.copy()
    coding_norms = np.linalg.norm(coding_dir, axis=0)
    coding_dir /= coding_norms
    D = Omega.shape[0]
    fact_adhoc_num = np.sum([coding_norms[i] ** 2 * coding_norms[j] ** 2 * (coding_dir[:, i] @ coding_dir[:, j]) ** 2 for i in range(D) for j in range(D)])
    fact_adhoc_den = np.sum([coding_norms[i] ** 2 * coding_norms[j] ** 2 for i in range(D) for j in range(D)])
    fact_adhoc = fact_adhoc_num / fact_adhoc_den

    return dim, c, snf, ssf, task_dim, zdim, fact_adhoc, eg 



#### PYTORCH IMPLEMENTATIONS


def theta_torch(x): 
    return torch.heaviside(x, values=torch.tensor([0.], device=device,dtype=dtype))

def emp_error_torch(points, latents, P, n_task=500, svc_analysis=True, max_bias = 0.25, min_task=100, dtype=dtype): 
    ''' 
    evaluate the generalization error empirically. torch implementation for gpu. 
    Args:
    - points: neural responses. ('x' variables)
    - latents: latent variables. ('z' variables) 
    - P: number of training samples to use 
    - n_task: number of T vectors to draw
    '''
    P_full, D = latents.shape 
    teach = torch.randn(D, n_task, device=device, dtype=dtype)
    # P x n_task labels 
    labels = torch.sign(latents @ teach) 
    # consider only problems with reasonably balanced labels
    fracp,fracm = torch.mean(1*(labels==1), dtype=dtype), torch.mean(1*(labels==-1), dtype=dtype)
    bias = torch.max(torch.stack([fracp, fracm],dim=0),dim=0) 
    valid = torch.abs(labels.mean(0)) < max_bias
    assert torch.sum(valid) > min_task, f"need more problems: {labels.mean(0)}"     
    labels = labels[:, valid]
    # split into train/test 
    errs = []
    idx = torch.randperm(P_full)
    train_idx, test_idx = idx[:P], idx[P:] 
    trainp, testp = points[train_idx], points[test_idx] 
    trainl, testl = labels[train_idx], labels[test_idx] 
    # n_task x N of readouts
    W = 1/P * torch.einsum('ms, mk -> sk', trainl, trainp)
    yhat = torch.sign(torch.einsum('sk, mk -> ms', W, testp))
    errs = torch.mean(theta(-testl * yhat) * 1.0, axis=0)
    if svc_analysis is True: 
        # cast labels to {1,0}
        trainl, testl = 1/2 * (1 + trainl), 1/2 * (1 + testl)
        val = (trainl.mean(0) != 1.0) & (trainl.mean(0) != 0.)
        if val.sum() < min_task: 
            err_svc = np.nan
        else: 
            trainl, testl = trainl[:, val].cpu().numpy(), testl[:, val].cpu().numpy()
            trainp, testp = trainp.cpu().numpy(), testp.cpu().numpy()
            cls = MultiOutputClassifier(LinearSVC(dual='auto'), n_jobs=-1)
            cls.fit(trainp, trainl)
            pred = cls.predict(testp) 
            err_svc = np.mean(pred != testl)
            # err_svc = cls.score(testp.cpu().numpy(), testl.cpu().numpy())
        return errs.mean(), err_svc, errs.std(), errs
    else:
        return errs.mean(), errs.std(), errs
    

