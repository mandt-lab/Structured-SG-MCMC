import warnings
warnings.filterwarnings("ignore")
import random
import os
import math
import numpy as np
import pandas as pd
from sklearn import preprocessing
import logging
from datetime import datetime
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import SGD, RMSprop
from torchvision import datasets, transforms
dtype = torch.cuda.FloatTensor
from datetime import datetime
import time
import argparse

from lvi.components import *
from lvi.optimizers import *
from lvi.utils import *
from lvi.mnist_utils import *
from lvi.convergence_criteria import *

import pickle

# create the log file 

folder_name = 'logs'

if not os.path.exists(folder_name):
    os.makedirs(folder_name)

# dd/mm/YY H:M:S
now = datetime.now()
dt_string = now.strftime("%d_%m_%Y_%H_%M_%S")
file_folder = folder_name+'/'+dt_string
if not os.path.exists(file_folder):
    os.mkdir(file_folder)
logger = log(file=dt_string+"/"+dt_string+".logs", path_to_folder = folder_name+'/', folder_name = folder_name)

# Construct the argument parser
ap = argparse.ArgumentParser()

# Add the arguments to the parser
ap.add_argument("--epochs", required=False, default = 300, type=int,
   help="Batch size for model")
ap.add_argument("--map_init", required=False, default = 'no', type=str,
   help="Whether or not we initialize the model with the MAP solution")
ap.add_argument("--lr", required=False, default=1e-2, type=float,
   help="constant learning rate for model")
ap.add_argument("--dataset", required=False, default = 'CIFAR10', type=str,
   help="choose dataset between wine and real_estate")
ap.add_argument("--grouping", required=False, default='permuted', type=str,
   help="Choose how to assign the groups. Randomly, Permuted, By_layer, None")
ap.add_argument("--dropout_prob", required=False, default=0.1 ,type=float,
   help="Choose the dropout probability for the model")
ap.add_argument("--vi_batch_size", required=False, default=16, type=int,
   help="Choose the vi_batch_size for the model")
ap.add_argument("--evaluation", required=False,default='yes', type=str,
   help="Choose whether or not to evaluate the model")
ap.add_argument("--conv", required=False, default='no',type=str,
   help="Choose whether or not to evaluate the model for convergence")
ap.add_argument("--gpu", required=False, default = '0', type=str,
   help="Choose GPU to use")
ap.add_argument("--batch_size", required=False, default = 128, type=int,
   help="Choose batch_size for the dataset")
ap.add_argument("--groups", required=False, default = 'all', type=str,
   help="Choose groups for the parameters")
ap.add_argument("--seed", required=False, default = 2, type=int,
   help="Choose seed number for the code")
ap.add_argument("--opt", required=False, default = 'psgld', type=str,
   help="Choose optimizer for the training of the model")

args, leftovers = ap.parse_known_args()

for arg in vars(args):
    logger.info("{} = {}".format(arg, getattr(args, arg)))
      
#set the seed
random.seed(args.seed)
np.random.seed(args.seed)
torch.manual_seed(args.seed)

## This is where the fun begins
if args.gpu=='cpu':
    device = 'cpu'
else:
    device = 'cuda:'+args.gpu

dev = torch.device(device)

trainloader, testloader, N = load_data(args.dataset, args.batch_size, dev)

dataiter = iter(testloader)
images, labels = dataiter.next()

print(images.shape)
print(labels.shape)

###### First we run the model with SGD in order to find the MAP solution. Then we run it again with SGLD
# in order to find the posterior distribution.

if args.map_init=='yes':
    
    if args.dataset == 'MNIST':
        run_sgd_map_model_mnist(trainloader, N, args.dataset, dev)
    else:   
        run_sgd_map_model(trainloader, N, args.dataset, dev)

### Here we run the main function. First we choose the arguements

if args.dataset == 'MNIST':
    if args.grouping is None:
        lvi_model = train_sgmcmc_model_mnist(trainloader, testloader, N, args.dataset, dev, args.groups, args.dropout_prob, args.lr, args.opt, args.vi_batch_size, args.epochs, logger, args.evaluation)
    else:
        lvi_model = train_s_sgmcmc_model_mnist(trainloader, testloader, N, args.dataset, dev, args.groups, args.dropout_prob, args.lr, args.opt, args.vi_batch_size, args.epochs, logger, args.evaluation)
        
else:
    if args.grouping is None:
        lvi_model = train_sgmcmc_model(trainloader, testloader, N, args.dataset, dev, args.groups, args.dropout_prob, args.lr, args.opt, args.vi_batch_size, args.epochs, logger, args.evaluation)
    else:
        lvi_model = train_s_sgmcmc_model(trainloader, testloader, N, args.dataset, dev, args.groups, args.dropout_prob, args.lr, args.opt, args.vi_batch_size, args.epochs, logger, args.evaluation)

if args.evaluation == 'yes':
    
    tmp_acc, tmp_ece = evaluate(lvi_model, testloader, N=len(valset))

    print("EVALUATION with 100 samples -> Loss: {}, CrossEntropy: {}, Accuracy: {}".format(sum(losses)/len(losses), sum(cross_losses)/len(cross_losses), tmp_acc))
    
    logger.info("EVALUATION with 100 samples -> Loss: {}, CrossEntropy: {}, Accuracy: {}".format(sum(losses)/len(losses), sum(cross_losses)/len(cross_losses), tmp_acc))
    
    logger.info("EVALUATION with 100 samples -> ECE: {}".format(tmp_ece))

if args.conv=='yes':
# we save the weights
    chains = []
    for name, chain in lvi_model.get_chains().items():
        chains.append(chain.view(chain.shape[0], -1).detach().cpu())
    w = torch.cat(chains, dim=-1).numpy()
    w = pd.DataFrame(w)
    print("W Shape is:",w.shape)

if args.conv=='yes':
    
    if args.dataset == 'MNIST':
        iac_time = calculate_IAC(w[10000:])
        ess_time = calculate_ESS(w[10000:])

        logger.info("IAC time = {}".format(iac_time))
        logger.info("ESS time = {}".format(ess_time))
    else:
        iac_time = calculate_IAC(w)
        ess_time = calculate_ESS(w)

        logger.info("IAC time = {}".format(iac_time))
        logger.info("ESS time = {}".format(ess_time))

