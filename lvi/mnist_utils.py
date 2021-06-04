import tensorflow_probability as tfp
import torch
from torch.distributions import Normal
from torch.optim.optimizer import Optimizer, required
import numpy as np
import pickle
import os
import logging
import warnings
warnings.filterwarnings("ignore")
import pandas as pd
import random
from sklearn import preprocessing
import seaborn as sns
import matplotlib.pyplot as plt
import pickle
import math
from datetime import datetime


import torch.nn as nn
import torch.nn.functional as F
from torch.optim import SGD, RMSprop
from torchvision import datasets, transforms

from lvi.components import *
from lvi.optimizers import *

random.seed(2)
np.random.seed(2)
torch.manual_seed(2)


def run_sgd_map_model_mnist(trainloader, N, dataset, dev):
    
    model_arch_args = dict(
        num_inputs=iter(trainloader).next()[0].shape[-1]*iter(trainloader).next()[0].shape[-2],
        num_outputs=10,
        num_layers=1,
        hidden_sizes=[10],
        activation_func=nn.ReLU,#nn.Tanh, #nn.ReLU,
        chain_length=5000,
        stochastic_biases=False,
        output_distribution="categorical",
        output_dist_const_params=dict(), #scale=1.0),
    )

    sgd_model_args = dict(
        group_by_layers=False,
        use_random_groups=False,
        use_permuted_groups=False,
        max_groups=None,
        dropout_prob=None,
        **model_arch_args,
    )
    
    sgd_model = FF_BNN(**sgd_model_args)
    sgd_model.initialize_optimizer(
        update_determ=True, 
        update_stoch=True, 
        lr=1e-8, #1e-5, 
        sgd=True, 
        sgld=False, 
        psgld=False,
    )
    
    pickle.dump(sgd_model_args, open("./mnist_sgd_model_params.pickle", "wb"))
        
    for n, t in sgd_model.tensor_dict.items():
        if isinstance(t, StochasticTensor):
            t.prior_dist.loc = t.prior_dist.loc.to(dev)
            t.prior_dist.scale = t.prior_dist.scale.to(dev)
            
    num_epochs = 1000
    criterion = torch.nn.CrossEntropyLoss()  # loss function

    for i in range(num_epochs):
        losses = []
        cross_losses = []
        accuracy = []

        for images, labels in trainloader:

            images = images.view(images.shape[0], -1)

            loss, y_pred,_ = sgd_model.training_step(
                batch=(images, labels),
                N=N,
                deterministic_weights=True,
                vi_batch_size=None,
            )
            losses.append(loss)

            cross_loss = criterion(y_pred.squeeze(0), labels)
            cross_losses.append(cross_loss)
            accuracy.append((torch.max(y_pred.squeeze(0),-1).indices == labels).sum().item() / labels.size(0))

        print("Iter {} / {}, Loss: {}, CrossEntropy: {}, Accuracy: {}".format(i+1, num_epochs, sum(losses), sum(cross_losses)/len(cross_losses), sum(accuracy)/len(accuracy)))
        
     #save the weights
 
    torch.save(sgd_model.state_dict(), "./sgd_nmnist_map.pt")     
        
#############################################################################################################################
 
def log(file, path_to_folder = 'logs/', folder_name = 'logs'):
    
    log_file = os.path.join(path_to_folder, file)
    
#     if not os.path.exists(folder_name):
#         os.makedirs(folder_name)
      
    if not os.path.isfile(log_file):
        open(log_file, "w+").close()

    console_logging_format = "%(levelname)s %(message)s"
    file_logging_format = "%(levelname)s: %(asctime)s: %(message)s"

    # configure logger
    logging.basicConfig(level=logging.INFO, format=console_logging_format)
    logger = logging.getLogger()
    
    # create a file handler for output file
    handler = logging.FileHandler(log_file)

    # set the logging level for log file
    handler.setLevel(logging.INFO)
    
    # create a logging format
    formatter = logging.Formatter(file_logging_format)
    handler.setFormatter(formatter)

    # add the handlers to the logger
    logger.addHandler(handler)

    return logger

#############################################################################################################################

def evaluate_mnist(lvi_model, testloader, N):
    criterion = torch.nn.CrossEntropyLoss()  # loss function
    losses = []
    cross_losses = []
    accuracy = []
    ece = []

    for images, labels in testloader:
        inner_cross_losses = []
        inner_accuracy = []
        inner_ece = []

        loss, y_pred = lvi_model.evaluate(batch=(images, labels),
                    N=N,
                    num_samples=100,
                    deterministic_weights=False)

        losses.append(loss)
        for j in range(y_pred.shape[0]):
            cross_loss = criterion(y_pred.squeeze(0)[j], labels)
            inner_cross_losses.append(cross_loss)
            inner_accuracy.append((torch.max(y_pred.squeeze(0)[j],-1).indices == labels).sum().item() / labels.size(0))
            inner_ece.append(tfp.stats.expected_calibration_error(num_bins=10, logits=y_pred.squeeze(0)[j].cpu().detach().numpy(), labels_true=labels.cpu().detach().numpy()))
            

            
            ece.append(sum(inner_ece)/len(inner_ece))
            accuracy.append(sum(inner_accuracy)/len(inner_accuracy))
            cross_losses.append(sum(inner_cross_losses)/len(inner_cross_losses))

    print("EVALUATION with 100 samples -> Loss: {}, CrossEntropy: {}, Accuracy: {}, ECE: {}".format(sum(losses)/len(losses), sum(cross_losses)/len(cross_losses), sum(accuracy)/len(accuracy), sum(ece)/len(ece)))
    return sum(accuracy)/len(accuracy), sum(ece)/len(ece)

#############################################################################################################################

def train_s_sgmcmc_model_mnist(trainloader, testloader, N, dataset, dev, args_groups, args_dropout_prob, args_lr, args_opt, args_vi_batch_size, args_epochs, logger, args_evaluation):

    sgd_model = FF_BNN(**pickle.load(open("./mnist_sgd_model_params.pickle", "rb")))
    sgd_model.load_state_dict(torch.load("./sgd_mnist_map.pt", map_location=dev))

    if args_groups=='all':
        num_stoch_params = 0
        for param in sgd_model.get_stochastic_params():
            param_size = 1
            for dim in param.shape:
                param_size *= dim
            num_stoch_params += param_size
        groups = num_stoch_params
        print("max_groups:", groups)
        logger.info("REAL max_groups = {}".format(groups))
    else:
        groups = int(args_groups)

    if args_dropout_prob is None:
        drop = None
    else:
        drop = float(args_dropout_prob)

    print("dropout:", drop)
    
    lvi_model_params = pickle.load(open("./mnist_sgd_model_params.pickle", "rb"))
        
    lvi_model_params["group_by_layers"] = False
    lvi_model_params["use_random_groups"] = False
    lvi_model_params["use_permuted_groups"] = True
    lvi_model_params["max_groups"] = groups
    lvi_model_params["dropout_prob"] = drop#None
    lvi_model_params["chain_length"] = 5000
    lvi_model_params["prior_std"] = 0.3 #1.0

    lvi_model_params["init_values"] = {k:v.theta_actual.data for k,v in sgd_model.tensor_dict.items()}
    del sgd_model

    lvi_model = FF_BNN(**lvi_model_params)

    if args_opt=='sghmc':
        lvi_model.initialize_optimizer(
            update_determ=False, 
            update_stoch=True, 
            lr=args_lr,
            sgd=False, 
            sgld=False, 
            psgld=False,
            sghmc=True,
        )
        print('INSIDE SGMC')
    else:
        lvi_model.initialize_optimizer(
            update_determ=False, 
            update_stoch=True, 
            lr=args_lr,
            sgd=False, 
            sgld=False, 
            psgld=True,
            sghmc=False,
        )
        print('INSIDE PSGLD/ELSE')

    lvi_model = lvi_model.to(dev)
    for n, t in lvi_model.tensor_dict.items():
        if isinstance(t, StochasticTensor):
            t.prior_dist.loc = t.prior_dist.loc.to(dev)
            t.prior_dist.scale = t.prior_dist.scale.to(dev)

    print("Before initialization: {}".format(lvi_model.num_samples_per_group))
    lvi_model.init_chains()
    print("After initialization: {}".format(lvi_model.num_samples_per_group))

    criterion = torch.nn.CrossEntropyLoss()  # loss function

    if args_vi_batch_size is None:
        vi_batch = None
    else:
        vi_batch = int(args_vi_batch_size)
    print("vi_batch:",vi_batch)

    print("learning rate:",args_lr)

    total_acc = []
    # start = time.time()
    for i in range(args_epochs):
        losses = []
        cross_losses = []
        accuracy = []

        start = time.time()

        for images, labels in trainloader:
            inner_cross_losses = []
            inner_accuracy = []
            
            images = images.view(images.shape[0], -1)

            if i < 1:
                loss, y_pred,_ = lvi_model.training_step(
                    batch=(images, labels),
                    N=N,
                    deterministic_weights=True,
                    vi_batch_size=vi_batch,
                )
            else:
                loss, y_pred,_ = lvi_model.training_step(
                    batch=(images, labels),
                    N=N,
                    deterministic_weights=False,
                    vi_batch_size=vi_batch,
                ) 

            losses.append(loss)

            with torch.no_grad():
                for j in range(y_pred.shape[0]):
                    cross_loss = criterion(y_pred.squeeze(0)[j], labels).item()
                    inner_cross_losses.append(cross_loss)
                    inner_accuracy.append((torch.max(y_pred.squeeze(0)[j],-1).indices == labels).sum().item() / labels.size(0))
            accuracy.append(sum(inner_accuracy)/len(inner_accuracy))
            cross_losses.append(sum(inner_cross_losses)/len(inner_cross_losses))

        print("Iter {} / {}, Loss: {}, CrossEntropy: {}, Accuracy: {}".format(i+1, args_epochs, sum(losses)/len(losses), sum(cross_losses)/len(cross_losses), sum(accuracy)/len(accuracy)))

        end = time.time()
        print('Elapsed time for the training:', end - start)
        if args_evaluation == 'yes':
            tmp_acc = evaluate(lvi_model, testloader, N=len(valset))
            total_acc.append(tmp_acc)
    logger.info("training_time_per_epoch = {}".format(end - start))
    logger.info("last_training_accuracy = {}".format(sum(accuracy)/len(accuracy)))
    logger.info("last_training_CrossEntropy = {}".format(sum(cross_losses)/len(cross_losses)))
    np.save(file_folder+'/dropout_lvi_acc_cifar10.npy', total_acc)
    
    return lvi_model

def train_sgmcmc_model_mnist(trainloader, testloader, N, dataset, dev, args_groups, args_dropout_prob, args_lr, args_opt, args_vi_batch_size, args_epochs, logger, args_evaluation):
    
    sgd_model = FF_BNN(**pickle.load(open("./mnist_sgd_model_params.pickle", "rb")))
    sgd_model.load_state_dict(torch.load("./sgd_mnist_map.pt", map_location=dev))

    if args_groups=='all':
        num_stoch_params = 0
        for param in sgd_model.get_stochastic_params():
            param_size = 1
            for dim in param.shape:
                param_size *= dim
            num_stoch_params += param_size
        groups = num_stoch_params
        print("max_groups:", groups)
        logger.info("REAL max_groups = {}".format(groups))
    else:
        groups = int(args_groups)

    if args_dropout_prob is None:
        drop = None
    else:
        drop = float(args_dropout_prob)

    print("dropout:", drop)
   
    lvi_model_params = pickle.load(open("./mnist_sgd_model_params.pickle", "rb"))
        
    lvi_model_params["group_by_layers"] = False
    lvi_model_params["use_random_groups"] = False
    lvi_model_params["use_permuted_groups"] = False
    lvi_model_params["max_groups"] = 1
    lvi_model_params["dropout_prob"] = drop#None
    lvi_model_params["chain_length"] = 5000
    lvi_model_params["prior_std"] = 0.3 #1.0

    lvi_model_params["init_values"] = {k:v.theta_actual.data for k,v in sgd_model.tensor_dict.items()}
    del sgd_model

    lvi_model = FF_BNN(**lvi_model_params)

    if args_opt=='sghmc':
        lvi_model.initialize_optimizer(
            update_determ=False, 
            update_stoch=True, 
            lr=args_lr,
            sgd=False, 
            sgld=False, 
            psgld=False,
            sghmc=True,
        )
        print('INSIDE SGMC')
    else:
        lvi_model.initialize_optimizer(
            update_determ=False, 
            update_stoch=True, 
            lr=args_lr,
            sgd=False, 
            sgld=False, 
            psgld=True,
            sghmc=False,
        )
        print('INSIDE PSGLD/ELSE')

    lvi_model = lvi_model.to(dev)
    for n, t in lvi_model.tensor_dict.items():
        if isinstance(t, StochasticTensor):
            t.prior_dist.loc = t.prior_dist.loc.to(dev)
            t.prior_dist.scale = t.prior_dist.scale.to(dev)

    print("Before initialization: {}".format(lvi_model.num_samples_per_group))
    lvi_model.init_chains()
    print("After initialization: {}".format(lvi_model.num_samples_per_group))

    criterion = torch.nn.CrossEntropyLoss()  # loss function

    if args_vi_batch_size is None:
        vi_batch = None
    else:
        vi_batch = int(args_vi_batch_size)
    print("vi_batch:",vi_batch)

    print("learning rate:",args_lr)

    total_acc = []
    # start = time.time()
    for i in range(args_epochs):
        losses = []
        cross_losses = []
        accuracy = []

        start = time.time()

        for images, labels in trainloader:
            inner_cross_losses = []
            inner_accuracy = []
            
            images = images.view(images.shape[0], -1)

            if i < 1:
                loss, y_pred,_ = lvi_model.training_step(
                    batch=(images, labels),
                    N=N,
                    deterministic_weights=True,
                    vi_batch_size=vi_batch,
                )
            else:
                loss, y_pred,_ = lvi_model.training_step(
                    batch=(images, labels),
                    N=N,
                    deterministic_weights=False,
                    vi_batch_size=vi_batch,
                ) 

            losses.append(loss)

            with torch.no_grad():
                cross_loss = criterion(y_pred.squeeze(0), labels)
                inner_cross_losses.append(cross_loss)
                inner_accuracy.append((torch.max(y_pred.squeeze(0),-1).indices == labels).sum().item() / labels.size(0))
            accuracy.append(sum(inner_accuracy)/len(inner_accuracy))
            cross_losses.append(sum(inner_cross_losses)/len(inner_cross_losses))

        print("Iter {} / {}, Loss: {}, CrossEntropy: {}, Accuracy: {}".format(i+1, args_epochs, sum(losses)/len(losses), sum(cross_losses)/len(cross_losses), sum(accuracy)/len(accuracy)))

        end = time.time()
        print('Elapsed time for the training:', end - start)
        if args_evaluation == 'yes':
            tmp_acc = evaluate(lvi_model, testloader, N=len(valset))
            total_acc.append(tmp_acc)
    logger.info("training_time_per_epoch = {}".format(end - start))
    logger.info("last_training_accuracy = {}".format(sum(accuracy)/len(accuracy)))
    logger.info("last_training_CrossEntropy = {}".format(sum(cross_losses)/len(cross_losses)))
    np.save(file_folder+'/dropout_lvi_acc_cifar10.npy', total_acc)
    
    return lvi_model


