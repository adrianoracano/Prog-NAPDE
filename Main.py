# -*- coding: utf-8 -*-
"""
Created on Fri Nov  3 17:18:07 2023

@author: alean
"""

from srcs import ModelClass
import argparse
import tensorflow as tf
import os
import numpy as np
import sys
import pickle
from utilities import SirHelperFunctions as shf
import random
import math
from srcs import plot_solutions
tfk = tf.keras
tfkl = tf.keras.layers
tfkl.Normalization(dtype='float64')

tf.keras.backend.set_floatx('float64')
seed = 25
random.seed(seed)
os.environ['PYTHONHASHSEED'] = str(seed)
np.random.seed(seed)
tf.random.set_seed(seed)
tf.compat.v1.set_random_seed(seed)


#######
# flags
#######
parser = argparse.ArgumentParser()
parser.add_argument('-t', '--train', help='train the network', action='store_true')
parser.add_argument('-i', '--iterations', help='override the number of iterations', type=int, default=0)
parser.add_argument('-ph', '--print-help', action='store_true')
parser.add_argument('-lm', '--load-model', help='load the model in the specified file before the training',default='')
parser.add_argument('-f', '--file', help='specify the file name. default: data.txt', default='data.txt')
parser.add_argument('-v', '--validate', help='use a validation set for the training', action='store_true')
parser.add_argument('-rp', '--random-plot', help='plot using randomly generated data', action='store_true')
parser.add_argument('-ptr', '--plot-train', help='plot using the training set', action='store_true')
parser.add_argument('-pte', '--plot-test', help='plot using the test set', action='store_true')
parser.add_argument('-o', '--overwrite', help='save and load the weights from the specified file', default='')
parser.add_argument('-sp', '--save-plots', help='save the plots in the specified file after the training', default='')
parser.add_argument('-sm', '--save-model', help='save the model file in the specified directory after the training', default='')

args = parser.parse_args()

###################
# dati del problema
###################
data_dict = {}
with open(args.file, 'r') as file:
    for line in file:
        line = line.strip()  # Remove leading/trailing whitespace
        if line:
            field, value = line.split(':')
            data_dict[field.strip()] = value.strip()
learning_rate = float(data_dict['learning_rate'])
t_max = float(data_dict['t_max'])
beta0 = float(data_dict['beta0'])
alpha = float(data_dict['alpha'])
training_steps = int(data_dict['training_steps'])
display_step = int(data_dict['display_step'])
S0 = float(data_dict['S0'])
I0 = float(data_dict['I0']  )
R0 = 1.0-I0-S0
S_inf = float(data_dict['S_inf'])
n_hidden = int(data_dict['n_hidden'])
display_weights = int(data_dict['display_weights'])
dataset_name = data_dict['dataset']
tau = float(data_dict['tau'])
n_input = 2

##################
# calcolo di b_ref
##################

b_ref = alpha * math.log(S0/S_inf) / (1.0 - S_inf)


###########################
# viene caricato il dataset
###########################
path_dataset = 'datasets/'+ dataset_name
try:
    with open(path_dataset, 'rb') as file:
        dataset,val_set,test_set = pickle.load(file)  # viene caricato il  dataset
    print('dataset', dataset_name, 'loaded...\n')
except FileNotFoundError:
    print('file',dataset_name,'not found...\n')
    sys.exit()
# ora nei dataset ci sono solo le temperature: bisogna calcolare i beta esatti

#################################
# vengono calcolati i beta esatti
#################################
beta_train, beta_val, beta_test = shf.compute_beta([dataset, val_set, test_set],\
                                                   beta0, t_max, tau, b_ref)

########################
# vengono calcolate le I
########################
I_train, I_val, I_test = shf.compute_I([beta_train, beta_val, beta_test],\
                                       t_max, alpha, [S0, I0, R0])
    



model = ModelClass.Model(n_hidden, learning_rate, b_ref, \
                   addDropout = False ,addBNorm = True, \
                   load_path = args.load_model)

network = ModelClass.NetworkForSIR(model, display_step, t_max, alpha)



if args.train:
    network.train(dataset, I_train, val_set, I_val, [S0, I0, R0], beta0, \
                  training_steps, display_weights, validate = True)

b_train_nn, I_train_nn = network.compute_beta_I(dataset, [S0, I0, R0], beta0)
b_val_nn, I_val_nn = network.compute_beta_I(val_set, [S0, I0, R0], beta0)
b_test_nn, I_test_nn = network.compute_beta_I(test_set, [S0, I0, R0], beta0)

if args.plot_train:
    plot_solutions.plot_beta_I(beta_train, I_train, b_train_nn, I_train_nn, \
                           "train", 5)
if args.plot_test:
    plot_solutions.plot_beta_I(beta_test, I_test, b_test_nn, I_test_nn, \
                           "test", 1)

if len(args.save_model)>0:
    model.save_model(args.save_model)


