import torch
import torch.nn as nn
from torchvision.models.feature_extraction import get_graph_node_names
from torchvision.models.feature_extraction import create_feature_extractor
import torch.nn as nn

from sklearn.multioutput import MultiOutputClassifier
from sklearn.svm import LinearSVC

import numpy as np 
from tqdm import tqdm

from utils import get_geometry, emp_error, theory_part


def draw_latents(P, D, alpha): 
    oms = (1+torch.arange(D))**(-alpha)
    samps = torch.randn(P, D) * torch.sqrt(oms)[None, :]
    return samps 

def gen_labels(z, num_tasks): 
    '''
    Generate task labels for the random shattering problem 
    '''
    P, D = z.shape 
    T = np.random.randn(D, num_tasks)
    return np.sign(z @ T), T 

def theta(x): 
    return (x>0)*1

def torch_theta(x):
    return 0.5 * (torch.sign(x) + 1.0) 


class GrowingMLP(nn.Module):
    ''' growing mlp architecture, possible with an extra projection head ''' 
    def __init__(self, input_size, growth_factor, num_layers):
        super(GrowingMLP, self).__init__()

        # Initialize the layers
        layers = []
        
        # Input layer
        layers.append(nn.Linear(input_size, int(input_size * growth_factor)))
        layers.append(nn.ReLU())

        # Intermediate hidden layers
        for i in range(1, num_layers - 1):
            in_size = int(input_size * growth_factor**i)
            out_size = int(input_size * growth_factor**(i + 1))
            layers.append(nn.Linear(in_size, out_size))
            layers.append(nn.ReLU())

        # Create the model
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)


class GrowingMLPTrained(nn.Module):
    ''' growing mlp architecture with batchnorm and extra projection head ''' 
    def __init__(self, input_size, growth_factor, num_layers, n_task):
        super(GrowingMLPTrained, self).__init__()

        # Initialize the layers
        layers = []
        
        # Input layer
        layers.append(nn.Linear(input_size, int(input_size * growth_factor)))
        layers.append(nn.BatchNorm1d(int(input_size * growth_factor)))
        layers.append(nn.ReLU())

        # Intermediate hidden layers
        for i in range(1, num_layers):
            in_size = int(input_size * growth_factor**i)
            out_size = int(input_size * growth_factor**(i + 1))
            layers.append(nn.Linear(in_size, out_size))
            layers.append(nn.BatchNorm1d(out_size))
            layers.append(nn.ReLU())

        # add a projection head  
        if n_task is not None: 
            layers.append(nn.Linear(out_size, n_task))

        # Create the model
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)


class MultiTaskMLP(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_tasks, num_hidden):
        super(MultiTaskMLP, self).__init__()

        # Initialize the layers
        layers = []
        
        # Input layer
        layers.append(nn.Linear(input_dim, hidden_dim))
        layers.append(nn.BatchNorm1d(hidden_dim))
        layers.append(nn.ReLU())

        # Intermediate hidden layers
        for i in range(1, num_hidden):
            layers.append(nn.Linear(hidden_dim, hidden_dim))
            layers.append(nn.BatchNorm1d(hidden_dim))
            layers.append(nn.ReLU())
        # linear readout head
        layers.append(nn.Linear(hidden_dim, num_tasks))

        # Create the model
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)
    
    
class MultiTaskMLP_tanh(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_tasks, num_hidden):
        super(MultiTaskMLP_tanh, self).__init__()

        # Initialize the layers
        layers = []
        
        # Input layer
        layers.append(nn.Linear(input_dim, hidden_dim))
        layers.append(nn.BatchNorm1d(hidden_dim))
        layers.append(nn.Tanh())

        # Intermediate hidden layers
        for i in range(1, num_hidden):
            layers.append(nn.Linear(hidden_dim, hidden_dim))
            layers.append(nn.BatchNorm1d(hidden_dim))
            layers.append(nn.Tanh())
        # linear readout head
        layers.append(nn.Linear(hidden_dim, num_tasks))

        # Create the model
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)


# Function to train the model
# def train_model(model, train_loader, val_loader, criterion, optimizer, outdir, num_epochs=10, device='cuda'):
#     val_losses, train_losses = np.zeros(num_epochs),  np.zeros(num_epochs)
#     for epoch in range(num_epochs):
#         running_loss = 0.0
#         running_err = 0.0
#         model.train() 
#         for inputs, labels in train_loader:
#             inputs, labels = inputs.to(device), labels.to(device) 
#             optimizer.zero_grad()
#             outputs = model(inputs)
#             loss = criterion(outputs.reshape(-1), labels.reshape(-1))
#             loss.backward()
#             optimizer.step()
#             running_loss += loss.item()
#             running_err += torch.mean(torch_theta(-(labels-1/2) * outputs)) 
#         print(f'Epoch {epoch + 1}/{num_epochs}, Training Loss: {running_loss / len(train_loader)}, Training err: {running_err / len(train_loader)}')
#         # Validation phase
#         model.eval()
#         val_loss = 0.0
#         task_std = 0.0 
#         with torch.no_grad():
#             for val_inputs, val_labels in val_loader:
#                 val_inputs, val_labels = val_inputs.to(device), val_labels.to(device)
#                 val_outputs = model(val_inputs)
#                 losses = torch_theta(-(val_labels-1/2) * val_outputs)
#                 val_loss += losses.mean() 
#                 # take the mean across batch dim and check for the std across tasks
#                 task_std += losses.mean(0).std() 
#         print(f'Epoch {epoch + 1}/{num_epochs}, Validation Error: {val_loss / len(val_loader)}, task std: {task_std/len(val_loader)}', flush=True)
#         val_losses[epoch], train_losses[epoch] = val_loss / len(val_loader), running_loss / len(train_loader)
#         # save model 
#         if (epoch /5).is_integer():
#             torch.save(model.state_dict(), outdir + f'/epoch_{epoch+1}.pth')
#     return val_losses, train_losses 


def train_model_online(model, train_loader, val_loader, criterion, optimizer, outdir, n_save=10, device='cuda'):
    num_step = len(train_loader) 
    step_iter = num_step // n_save 
    targ_late = np.logspace(1, np.log10(num_step), n_save, dtype=int)
    targ_early = np.array([1,2,3,4,5])
    targ_is = np.concatenate([targ_early, targ_late])
    print("target indices = ", targ_is,flush=True)
    val_losses, train_losses = np.zeros(len(targ_is)),  np.zeros(len(targ_is))
    running_loss = 0.0
    running_err = 0.0
    j=0
    batch_size = train_loader.batch_size
    for i, (inputs, labels) in enumerate(train_loader):
        model.train() 
        inputs, labels = inputs.to(device), labels.to(device) 
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs.reshape(-1), labels.reshape(-1))
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
        running_err += torch.mean(torch_theta(-(labels-1/2) * outputs)) 
        
        if (i+1) in targ_is : 
            print(f'iter {i+1}, Training Loss: {running_loss / len(train_loader)}, Training err: {running_err / len(train_loader)}')
            # Validation phase
            model.eval()
            val_loss = 0.0
            task_std = 0.0 
            with torch.no_grad():
                for val_inputs, val_labels in val_loader:
                    val_inputs, val_labels = val_inputs.to(device), val_labels.to(device)
                    val_outputs = model(val_inputs)
                    losses = torch_theta(-(val_labels-1/2) * val_outputs)
                    val_loss += losses.mean() 
                    # take the mean across batch dim and check for the std across tasks
                    task_std += losses.mean(0).std() 
            print(f'iter {i+1}, Validation Error: {val_loss / len(val_loader)}, task std: {task_std/len(val_loader)}', flush=True)
            val_losses[j], train_losses[j] = val_loss / len(val_loader), running_loss / len(train_loader)
            p = batch_size * (i+1) 
            j += 1 
            # save model 
            torch.save({'weights' : model.state_dict(), 'p' : p}, outdir + f'/iter_{i+1}.pth')
    return val_losses, train_losses 





#### HEBBIAN EVALUATION 
            


# def get_geometry(Psi, Phi, Omega): 
#     PR = np.trace(Psi)**2 / np.trace(Psi@Psi.T)
#     corr = np.trace(Phi@Phi.T) / np.trace(Omega) / np.trace(Psi)
#     # align = np.trace(Phi.T @ Psi @ Phi) / np.trace(Psi) / np.trace(Phi@Phi.T)
#     # new terms
#     Omega_inv = np.linalg.inv(Omega)
#     H = Psi - Phi @ Omega_inv @ Phi.T 
#     fact = np.trace(Phi.T @ Phi)**2 / np.trace(Omega) / np.trace(Phi.T @ Phi @ Omega_inv @ Phi.T @ Phi)
#     sna = np.trace(Phi.T @ Phi)**2 / np.trace(Omega) / np.trace(Phi.T @ H @ Phi)
#     return PR, corr, fact, sna
# # import numpy.linalg as npl
# # def get_geometry(Psi, Phi, Omega, P=100): 
# #     PR = np.trace(Psi)**2 / np.trace(Psi@Psi.T)
# #     corr = np.trace(Phi@Phi.T) / np.trace(Omega) / np.trace(Psi)
# #     align = np.trace(Phi.T @ Psi @ Phi) / np.trace(Psi) / np.trace(Phi@Phi.T)
# #     fact = np.trace(Omega) * np.trace(Phi.T @ Phi @ npl.inv(Omega) @ Phi.T @ Phi) / (np.trace(Phi @ Phi.T) ** 2)
# #     H = Psi - Phi @ npl.inv(Omega) @ Phi.T
# #     sna = np.trace(Omega) * np.trace(Phi.T @ H @ Phi) / (np.trace(Phi @ Phi.T)) ** 2
# #     Phi_hat = Phi / npl.norm(Phi, axis=0)
# #     D = Omega.shape[0]
# #     fact_adhoc = (np.sum((Phi_hat.T @ Phi_hat) ** 2) - D) / (D * (D - 1))
# #     print(Phi_hat.T @ Phi_hat)
# #     term_1 = np.pi / (corr ** 2 * PR * P)
# #     term_2 = 2 * (sna + fact)
# #     return PR, corr,  1/fact, 1/sna

# def theory_part(Psi, Phi, Omega, P):
#     T1 = np.pi/P * np.trace(Psi@Psi.T) * np.trace(Omega) 
#     T2 = 2 * (np.trace(Phi.T @ Psi @ Phi) - np.trace(Phi.T@Phi)**2/np.trace(Omega)) 
#     T3 = np.sqrt(2) * np.trace(Phi@Phi.T)
#     return 1/np.pi * np.arctan(np.sqrt(np.trace(Omega)*(T1+T2))/T3), get_geometry(Psi, Phi, Omega) 

# def emp_error(points, latents, P, return_svm, n_task=500, max_bias = 0.25, min_task=100): 
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
#     if return_svm is True: 
#         # cast labels to {1,0}
#         trainl, testl = 1/2 * (1 + trainl), 1/2 * (1 + testl)
#         cls = MultiOutputClassifier(LinearSVC(dual='auto'), n_jobs=-1)
#         cls.fit(trainp, trainl)
#         pred = cls.predict(testp)
#         err_svc = np.mean(pred != testl)
#     else: 
#         err_svc = np.nan 
#     return errs.mean(), err_svc, errs.std(), errs, 

# def emp_error(points, latents, P, n_task=300, teach=None): 
#     P_full, D = latents.shape 
#     if teach is None: 
#         teach = np.random.randn(D, n_task)
#     # P x n_task labels 
#     labels = np.sign(latents @ teach) 
#     errs = []
#     idx = np.arange(P_full)
#     train_idx = np.random.choice(idx, replace=False, size=(P,))
#     # sample train and test set 
#     test_idx = idx[~np.isin(idx, train_idx)]
#     trainp, testp = points[train_idx], points[test_idx] 
#     trainl, testl = labels[train_idx], labels[test_idx] 
#     # n_task x N of readouts
#     W = 1/P * np.einsum('ms, mk -> sk', trainl, trainp)
#     proj = np.einsum('sk, mk -> ms', W, testp)
#     yhat = np.sign(proj)
#     errs = np.mean(theta(-testl * yhat), axis=0)
#     return errs.mean(), errs.std(), errs, testl * proj
    
def get_errors(x, z, P, Pall, return_svm, nrep=3):
    '''Pall is total samples num. and P is the number of training samples for the hebb rule'''
    x,z = x-x.mean(0,keepdim=True), z-z.mean(0,keepdim=True)
    Psi = x.T @ x / Pall 
    Phi = x.T @ z / Pall 
    Omega = z.T @ z / Pall 
    x,z = x.detach().cpu().numpy(), z.detach().cpu().numpy()
    Psi, Phi, Omega = Psi.detach().cpu().numpy(), Phi.detach().cpu().numpy(), Omega.detach().cpu().numpy()
    seedos = np.random.randint(low=0,high=60_000, size=(nrep,))
    # err_the, geom = theory_part(Psi, Phi, Omega, P)
    err_the = theory_part(Psi, Phi, Omega, P)
    dim, c, snf, ssf, _, _, fact_adhoc, _ = get_geometry(Psi, Phi, Omega, P)
    geom = (dim, c, ssf, snf, fact_adhoc)
    errs_emp = np.zeros(nrep) 
    errs_svm = np.zeros(nrep) 
    for n in range(nrep): 
        np.random.seed(seedos[n])
        m, s, *_ = emp_error(x,z,P,svc_analysis=return_svm)
        errs_emp[n] = m 
        if return_svm == True: 
            errs_svm[n] = s
    errs_std = errs_emp.std() 
    errs_mu = errs_emp.mean() 
    err_svm_mu = errs_svm.mean() 
    err_svm_std = errs_svm.std() 
    return err_the, errs_mu, errs_std, err_svm_mu, err_svm_std, geom


def get_eg_rand(gen_model, num_layers, D, alpha, device, Pall=1000, Ptarg=300, nonlin='relu',return_svm=True):
    # layer names
    rand_layer_names = [[f'linear {i}', f'{nonlin} {i}'] for i in range(1,num_layers)]
    rand_layer_names = np.array(rand_layer_names).flatten()
    rand_layer_names = np.concatenate([['input'], rand_layer_names])
    # sample latents and extract feat. 
    nodes=get_graph_node_names(gen_model)[0]
    extr = create_feature_extractor(gen_model, nodes)
    z = draw_latents(Pall,D,alpha).to(device)
    out = extr(z)
    # init errs
    errs_the_rand = np.zeros(len(out.keys()))
    errs_emp_rand = np.zeros((len(out.keys()), 2))
    errs_svm_rand = np.zeros((len(out.keys()), 2))
    geoms_rand = np.zeros((len(out.keys()), 5))
    # it thru layers                 
    for i, val in enumerate(out.values()): 
        t, em, es, emsvm, essvm, geom = get_errors(val,z,Ptarg,Pall,return_svm=return_svm)
        errs_the_rand[i], errs_emp_rand[i], errs_svm_rand[i], geoms_rand[i] = t, (em,es), (emsvm, essvm), geom
    return rand_layer_names, errs_the_rand, errs_emp_rand, errs_svm_rand, geoms_rand

def get_eg_trained(model,gen_model, num_hidden, D, alpha, device, Pall=1000, Ptarg=300, nonlin='relu', includebnorm=False, return_svm=True):
    ''' 
    evaluate a trained model. The key difference is that hold out points have to be pushed through the gen_model first! 
    '''
    trained_layer_names = [[f'linear {i}', f'b.norm {i}', f'{nonlin} {i}'] for i in range(1,num_hidden+1)]
    trained_layer_names = np.array(trained_layer_names).flatten()
    #### MAYBE THIS LAST ONE SHOULD BE ELIMINATED??
    trained_layer_names= np.concatenate([['input'], trained_layer_names, [f'linear {num_hidden+1}']])
    # node names 
    nodes=get_graph_node_names(model)[0]
    # nodes=nodes[
    print('nodes = ', nodes) 
    print('tlayernames = ', trained_layer_names)
    if includebnorm is False: 
        # get ids of non-bn layers
        idxs = np.array(['b.norm' not in x for x in trained_layer_names])
        nodes = list(np.array(nodes)[idxs])
        trained_layer_names = trained_layer_names[idxs]
    model.eval() 
    extr = create_feature_extractor(model, nodes)
    # draw new latents and run 
    znew = draw_latents(Pall,D,alpha).to(device)
    print(gen_model(znew))
    out_new = extr(gen_model(znew))
    # initialize errors
    errs_the_train = np.zeros(len(out_new.keys()))
    errs_emp_train  = np.zeros((len(out_new.keys()), 2))
    errs_svm_train = np.zeros((len(out_new.keys()), 2))
    geoms_train  = np.zeros((len(out_new.keys()), 5))
    for i, val in tqdm(enumerate(out_new.values()), total=len(out_new.values())): 
        t, em, es, emsvm, essvm, geom = get_errors(val,znew,Ptarg,Pall,return_svm=return_svm)
        errs_the_train[i], errs_emp_train[i], errs_svm_train[i], geoms_train[i] = t, (em,es), (emsvm, essvm), geom 
    return trained_layer_names, errs_the_train, errs_emp_train, errs_svm_train, geoms_train 


def get_eg_thru_training(weight_paths, model, gen_model, num_hidden, D, alpha, device, Pall=1000, Ptarg=300, includebnorm=False, nonlin='relu',return_svm=False):
    errs_t, errs_e, errs_svm, geoms_tr, ps = [], [], [], [], [] 
    for path in sorted(weight_paths): 
        chck = torch.load(path, weights_only=True)
        weights, p = chck['weights'], chck['p']
        model.load_state_dict(weights)
        trained_layer_names, errs_the_train, errs_emp_train, errs_svm_train, geoms_train  = get_eg_trained(model, gen_model, num_hidden, D, alpha, device, Pall=Pall, Ptarg=Ptarg, includebnorm=includebnorm, nonlin='relu', return_svm=return_svm)
        errs_t.append(errs_the_train), errs_e.append(errs_emp_train), errs_svm.append(errs_svm_train), geoms_tr.append(geoms_train)
        ps.append(p)
    return trained_layer_names, np.stack(errs_t), np.stack(errs_e), np.stack(errs_svm), np.stack(geoms_tr), ps 