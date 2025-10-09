#!/usr/bin/env python
#SBATCH -p gpu
#SBATCH --gpus=1
#SBATCH --cpus-per-gpu=10
#SBATCH --mem=250G
#SBATCH --time=0-3

import os 
import sys 
from glob import glob 
base = '/mnt/home/awakhloo/ceph/reprod_population_geom_opt_coding/code/'
sys.path.append(base)

import numpy as np
import torch

#### MAINTEXT
np.random.seed(3487563)
torch.manual_seed(864523)
nonlin='relu' 
alpha=0.2


# #### SM: FANOUT 
# np.random.seed(42)
# torch.manual_seed(534534)
# nonlin='relu_fanout' 
# alpha=0.2
# growth_factor_train = 1.5

# ### SM: TANH
# np.random.seed(235235)
# torch.manual_seed(456466)
# nonlin='tanh'
# alpha=0.2


import torch.nn as nn
from torchvision.models.feature_extraction import get_graph_node_names
from torchvision.models.feature_extraction import create_feature_extractor
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, random_split
from sklearn.multioutput import MultiOutputClassifier
from sklearn.svm import LinearSVC


import matplotlib.pyplot as plt 
from tqdm import tqdm
from scipy.special import erfc
from glob import glob
import re 
import MLP_models as mlp 


device='cuda'
outdir = '/mnt/home/awakhloo/ceph/reprod_population_geom_opt_coding/results'



# hyperparams for training data generation 
Pall = 500_000 # num train pts 
D = 40 

output_size = 10  #
growth_factor = 2.
# num_layers = 4

# hyperparams for train model 
output_dim = 1  # Output dimension for each task
num_hidden=4
batch_size = 512  
learning_rate = 0.003 
n_task = 500
train_ratio = 0.9  # 80% for training, 20% for testing
# n_epoch = 40
n_save = 10 # number of training points to save 
h_scaling = 3 # hidden layer size / input size


# make outpath
outpath = outdir + f'/mlp_exps_P_{Pall}_nonlin_{nonlin}_alpha_{alpha}'
os.makedirs(outpath + '/model_weights', exist_ok=True)

# Create the model
gen_model = mlp.GrowingMLP(D, growth_factor, num_hidden).to(device)
# sample latents
z = mlp.draw_latents(Pall,D,alpha)
points= gen_model(z.to(device)) 
print(gen_model,flush=True)

# use the outputs of the random network
input_dim =  points.shape[1] 
hidden_dim = h_scaling*input_dim

# Create a random dataset 
points = points.detach()
labels, T = mlp.gen_labels(z, n_task)
# shift labels to 0,1 rather than -1,1
labels = (labels + 1)/2
# Create DataLoader
dataset = TensorDataset(points, labels.float())  # Assuming float labels for BCE loss
num_train_samples = int(train_ratio * len(dataset))
num_val_samples = len(dataset) - num_train_samples
train_set, val_set = random_split(dataset, [num_train_samples, num_val_samples])

# Create DataLoader for train, validation, and test sets
train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False)

# Initialize the model, loss function, and optimizer
if nonlin == 'relu' :
    model = mlp.MultiTaskMLP(input_dim, hidden_dim, output_dim, n_task, num_hidden)
elif nonlin == 'tanh' : 
    model = mlp.MultiTaskMLP_tanh(input_dim, hidden_dim, output_dim, n_task, num_hidden)
elif nonlin == 'relu_fanout': 
    model = mlp.GrowingMLPTrained(input_dim, growth_factor_train, num_hidden, n_task).to(device) 

model = model.to(device)
criterion = nn.BCEWithLogitsLoss()  # Binary Cross Entropy loss
optimizer = optim.Adam(model.parameters(), lr=learning_rate)
print(model) 

val_losses, train_losses = mlp.train_model_online(model,
                                          train_loader,
                                          val_loader,
                                          criterion,
                                          optimizer,
                                          outpath + '/model_weights',
                                          n_save=n_save,
                                          device=device)

# Eval rand and trained nn 
rand_layer_names, errs_the_rand, errs_emp_rand, errs_svm_rand, geoms_rand  = mlp.get_eg_rand(gen_model, 
                                                                              num_hidden,
                                                                              D,
                                                                              alpha,
                                                                              device,
                                                                             nonlin=nonlin) 

trained_layer_names, errs_the_train, errs_emp_train, errs_svm_train, geoms_train  = mlp.get_eg_trained(
                            model,
                            gen_model, 
                            num_hidden, 
                            D, 
                            alpha,
                            device,
                            nonlin=nonlin.replace("_fanout", ""))


# save rand model weights
torch.save(gen_model.state_dict(), outpath + '/model_weights/base_model.pth')
# save Eg and geom curves
results = {'rand_names' : rand_layer_names,
           'errs_t_rand' :  errs_the_rand, 
           'errs_e_rand' : errs_emp_rand, 
           'geoms_rand' : geoms_rand,
           'tr_names' : trained_layer_names, 
           'errs_t_tr' : errs_the_train, 
           'errs_e_tr' : errs_emp_train, 
           'geoms_tr' : geoms_train,
           'val_losses' : val_losses, 
           'train_losses' : train_losses, 
           'errs_svm_rand' : errs_svm_rand,
           'errs_svm_train' : errs_svm_train,
           }

#### DYNAMICS OF GEOMETRY AND ERROR
weight_paths =glob(outpath + '/model_weights/iter_*.pth')
_, errs_t, errs_e, errs_svm, geoms_tr, ps = mlp.get_eg_thru_training(weight_paths,
                                                   model,
                                                   gen_model,
                                                   num_hidden,
                                                   D,
                                                   alpha,
                                                   device,
                                                   nonlin=nonlin.replace("_fanout", ""))
results['dyn_err_t'] = errs_t
results['dyn_err_e'] = errs_e
results['dyn_geom'] = geoms_tr
results['ps'] = ps
results['dyn_errs'] = errs_svm
np.save(outpath + '/eg_geom_results.npy', np.array(results))
