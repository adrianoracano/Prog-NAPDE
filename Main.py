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
from matplotlib import pyplot as plt
from srcs import plot_solutions
import shutil
tfk = tf.keras
tfkl = tf.keras.layers
tfkl.Normalization(dtype='float64')

tf.keras.backend.set_floatx('float64')
seed = 29032024
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
parser.add_argument('-tc', '--test-case', help='start a test case', action='store_true')
parser.add_argument('-bl', '--beta-log', help='start a beta-log test-case', action='store_true', default = False)

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
num_layers = int(data_dict['num_layers'])

##################
# calcolo di b_ref
##################

b_ref = alpha * math.log(S0/S_inf) / (1.0 - S_inf)

###########################
# viene caricato il dataset
###########################
n_giorni = None
date_train = None
date_val = None
path_dataset = 'datasets/'+ dataset_name
try:
    with open(path_dataset, 'rb') as file:
        if args.test_case:
            dataset,val_set,test_set,beta0_train,beta0_val,beta0_test = pickle.load(file)  # viene caricato il  dataset
        elif args.beta_log:
            # vengono presi gli infetti (ricostruiti col beta-log), il beta-log e le temperature (+ zone, ...)
            I_train, I_val, dataset, val_set, beta_train, beta_val, n_giorni, date_train, date_val = pickle.load(file)
            t_max = n_giorni/30
        else:
            # Se si vogliono utilizzare dati reali vengono caricati gli infetti
            # e dati vari, come le temperature e i valori di beta0
            I_train, I_val, dataset, val_set, beta0_train, beta0_val = pickle.load(file)
    print('dataset', dataset_name, 'loaded...\n')
except FileNotFoundError:
    print('file',dataset_name,'not found...\n')
    sys.exit()
# ora nei dataset ci sono solo le temperature: bisogna calcolare i beta esatti

#####################
# qualche print utile
#####################
print('data:\n')
print('S0: ', S0, ', I0: ', I0, ', S_inf: ', S_inf)
print('bet_ref: ', b_ref)
print('K:', dataset.shape[0], ', K_val:', val_set.shape[0])
print('t_max (in mesi) : ', t_max, ', N: ', dataset.shape[1], 'dt: ', round(1/dataset.shape[1], 6) )
print('layers: ', num_layers)
print('hidden neurons per layer: ', n_hidden, '\n')

# se si vuole fare il training su dati sintetici bisogna calcolare i beta e 
# gli infetti:

if args.test_case and not args.beta_log:
    #################################
    # vengono calcolati i beta esatti
    #################################
    beta_train, beta_val, beta_test = shf.compute_beta([dataset, val_set, test_set],\
                                                       [beta0_train, beta0_val, beta0_test],\
                                                       t_max, tau, b_ref)
    ########################
    # vengono calcolate le I
    ########################
    I_train, I_val, I_test = shf.compute_I([beta_train, beta_val, beta_test],\
                                           t_max, alpha, [S0, I0, R0])
        
# il numero di neuroni in input vengono dedotti dalla shape del dataset

if len(dataset.shape) == 2:
    n_input = 2
else:
    n_input = dataset.shape[2]+1

use_keras = False
addDropout = False

#use_TensorBoard = False
#tf.summary.trace_on(graph=True, profiler=True)

model = ModelClass.Model(n_input, n_hidden, learning_rate, b_ref, \
                   addDropout = False ,addBNorm = addDropout, \
                   load_path = args.load_model,
                         use_keras = use_keras,
                         numLayers = num_layers
                         #,use_TensorBoard=use_TensorBoard
                         )

network = ModelClass.NetworkForSIR(model, display_step, t_max, alpha)

# se si utilizza il test-case beta-log vengono presi da beta_train, beta_val i
# valori iniziali beta0_train, beta0_val
if args.beta_log:
    beta0_train = beta_train[:, 0]
    beta0_val = beta_val[:, 0]

if args.train:
    loss_train, loss_val, it = network.train(dataset, I_train, val_set, I_val, \
                                             beta0_train, beta0_val, \
                                             training_steps, display_weights, validate = True)

save_folder = str(model.numLayers) + 'x' + str(model.n_hidden) + 'neu_' + str(model.n_iter) + 'iter_' + dataset_name[:-4]
saved_model_path = args.save_model + '/' + save_folder

if len(args.save_model)>0:
    model.save_model(saved_model_path)

b_train_nn, I_train_nn = network.compute_beta_I(dataset, I_train[:, 0], beta0_train)
b_val_nn, I_val_nn = network.compute_beta_I(val_set, I_val[:, 0], beta0_val)
if args.test_case:
    b_test_nn, I_test_nn = network.compute_beta_I(test_set, I_test[:, 0], beta0_test)

########################
# plot di beta e infetti
########################

saved_plots_path = ''
if len(args.save_plots) > 0:
    saved_plots_path = args.save_plots + '/' + save_folder

# se è un test case allora si conoscono anche i beta esatti che quindi vengono
# plottati

got_beta_for_plot = args.test_case or args.beta_log #test case e beta log non devono essere messi insieme

if args.beta_log:
    if args.plot_train:
        plot_solutions.plot_beta_I_2(I_train_nn, b_train_nn, I_train, beta_train, \
                                     "train", 1, save_plots=saved_plots_path, n_giorni = n_giorni, date = date_train)
    if args.plot_test:
        plot_solutions.plot_beta_I_2(I_val_nn, b_val_nn, I_val, beta_val, \
                                     "val", 1, save_plots=saved_plots_path, n_giorni = n_giorni, date = date_val)

if args.test_case:
    if args.plot_train:
        plot_solutions.plot_beta_I_2(I_train_nn, b_train_nn, I_train, beta_train, \
                                     "train", 1, save_plots=saved_plots_path)
    if args.plot_test:
        plot_solutions.plot_beta_I_2(I_val_nn, b_val_nn, I_val, beta_val, \
                                     "val", 1, save_plots=saved_plots_path)

# se si utilizzano dati reali i beta esatti sono sconosciuti, quindi vengono
# fatti i plot solo degli infetti reali, degli infetti calcolati dalla rete, e
# dei beta calcolati dalla rete

if args.plot_train and not got_beta_for_plot:
    plot_solutions.plot_beta_I(I_train_nn, b_train_nn, I_train, beta_train, \
                                 "train", 1, save_plots=saved_plots_path)
if args.plot_test and not got_beta_for_plot:
    plot_solutions.plot_beta_I(I_test_nn, b_test_nn, I_test, beta_test, \
                                 "test", 1, save_plots=saved_plots_path)
#################
# plot della loss
#################

if args.train:
    plt.figure()
    plt.plot(display_step*np.arange(0, it), loss_train[0:it])
    plt.plot(display_step*np.arange(0, it), loss_val[0:it])
    plt.legend(["loss train", "loss val"])
    plt.title("loss evolution")
    if len(args.save_plots) > 0:
        print("Saving the loss plot in " + saved_plots_path + "...\n")
        path = "./" + saved_plots_path;
        if not os.path.exists(path):
            os.makedirs(path)
        filepath2 = path + "/loss_plot.png";
        plt.savefig(fname=filepath2)

    plt.figure()
    plt.semilogy(display_step*np.arange(0, it), loss_train[0:it])
    plt.semilogy(display_step*np.arange(0, it), loss_val[0:it])
    plt.legend(["loss train", "loss val"])
    plt.title("loss evolution (semilog scale)")
    if len(args.save_plots) > 0:
        print("Saving the loss plot in " + saved_plots_path + "...\n")
        path = "./" + saved_plots_path;
        if not os.path.exists(path):
            os.makedirs(path)
        filepath2 = path + "/loss_plot_semilog.png";
        plt.savefig(fname=filepath2)
    #plt.show()
    plt.close()


if len(args.save_plots) > 0:
    shutil.copy('data.txt', saved_plots_path)
    shutil.copy('utilities\data-for-GenerateDataset.txt', saved_plots_path)
    print('saved data')

if len(args.save_model) > 0:
    shutil.copy('data.txt', saved_model_path)
    shutil.copy('utilities\data-for-GenerateDataset.txt', saved_model_path)
    print('saved data')

    


