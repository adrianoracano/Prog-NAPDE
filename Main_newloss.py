# -*- coding: utf-8 -*-
"""
Created on Fri Nov  3 17:18:07 2023

@author: alean
"""

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
from srcs import ModelClass_newloss
from srcs import plot_solutions_newloss


tfk = tf.keras
tfkl = tf.keras.layers
tfkl.Normalization(dtype='float64')

tf.keras.backend.set_floatx('float64')
seed = 49
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
parser.add_argument('-lm', '--load-model', help='load the model in the specified file before the training', default='')
parser.add_argument('-f', '--file', help='specify the file name. default: data.txt', default='data.txt')
parser.add_argument('-v', '--validate', help='use a validation set for the training', action='store_true')
parser.add_argument('-rp', '--random-plot', help='plot using randomly generated data', action='store_true')
parser.add_argument('-ptr', '--plot-train', help='plot using the training set', action='store_true')
parser.add_argument('-pte', '--plot-test', help='plot using the test set', action='store_true')
parser.add_argument('-o', '--overwrite', help='save and load the weights from the specified file', default='')
parser.add_argument('-sp', '--save-plots', help='save the plots in the specified file after the training', default='')
parser.add_argument('-sm', '--save-model', help='save the model file in the specified directory after the training',
                    default='')
parser.add_argument('-tc', '--test-case', help='start a test case', action='store_true')

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
I0 = float(data_dict['I0'])
R0 = 1.0 - I0 - S0
S_inf = float(data_dict['S_inf'])
n_hidden = int(data_dict['n_hidden'])
display_weights = int(data_dict['display_weights'])
dataset_name = data_dict['dataset']
tau = float(data_dict['tau'])

##################
# calcolo di b_ref
##################

b_ref = alpha * math.log(S0 / S_inf) / (1.0 - S_inf)

#####################
# flag del modello
#####################

learn_alpha = True #la rete impara alpha, bisogna scegliere il dataset con anche i rimossi
use_keras = False #usa keras
addDropout = False #(solo per use_keras = True)
addBnorm = True #(solo per use_keras = True)
no_SIR = False  #(funziona male) traina senza usare il sir
use_dI = False #(solo per learn_alpha = True) aggiunge la derivata degli infetti nella loss(ci vuole molto più tempo)
MSE_loss = True #(solo per learn_alpha = True) usa la loss con l'errore quadratico medio invece del modulo
plot = False #plotta anche invece di salvare e basta
saving_step = 500 #ogni quanti step salvare il modello (può anche essere None)

###########################
# viene caricato il dataset
###########################
path_dataset = 'datasets/' + dataset_name
try:
    with open(path_dataset, 'rb') as file:
        if args.test_case:
            dataset, val_set, test_set, beta0_train, beta0_val, beta0_test = pickle.load(
                file)  # viene caricato il  dataset
        else:
            # Se si vogliono utilizzare dati reali vengono caricati gli infetti
            # e dati vari, come le temperature e i valori di beta0
            if not learn_alpha:
                I_train, I_val, dataset, val_set, beta0_train, beta0_val = pickle.load(file)
            else:
                I_train, I_val, R_train, R_val, dataset, val_set, beta0_train, beta0_val = pickle.load(file)

    print('dataset', dataset_name, 'loaded...\n')
except FileNotFoundError:
    print('file', dataset_name, 'not found...\n')
    sys.exit()
# ora nei dataset ci sono solo le temperature: bisogna calcolare i beta esatti

#####################
# qualche print utile
#####################
print('data:\n')
print('S0: ', S0, ', I0: ', I0, ', S_inf: ', S_inf)
print('bet_ref: ', b_ref)
print('K:', dataset.shape[0], ', K_val:', val_set.shape[0])
print('t_max: ', t_max, ', N: ', dataset.shape[1], 'dt: ', round(1 / dataset.shape[1], 6))
print('hidden neurons: ', n_hidden, '\n')

# se si vuole fare il training su dati sintetici bisogna calcolare i beta e
# gli infetti:

if args.test_case:
    #################################
    # vengono calcolati i beta esatti
    #################################
    beta_train, beta_val, beta_test = shf.compute_beta([dataset, val_set, test_set], \
                                                       [beta0_train, beta0_val, beta0_test], \
                                                       t_max, tau, b_ref)
    ########################
    # vengono calcolate le I
    ########################
    I_train, I_val, I_test = shf.compute_I([beta_train, beta_val, beta_test], \
                                           t_max, alpha, [S0, I0, R0])

# il numero di neuroni in input vengono dedotti dalla shape del dataset

if len(dataset.shape) == 2:
    n_input = 2
else:
    n_input = dataset.shape[2] + 1
if learn_alpha:
    n_input = n_input + 1


model = ModelClass_newloss.Model(n_input, n_hidden, learning_rate, b_ref, \
                         addDropout=addDropout,
                         addBNorm=addBnorm, \
                         load_path=args.load_model,
                         use_keras=use_keras,
                         no_SIR = no_SIR,
                         use_dI = use_dI,
                         learn_alpha = learn_alpha,
                         MSE_loss = MSE_loss)

network = ModelClass_newloss.NetworkForSIR(model, display_step, t_max, alpha)

if args.train:
    loss_train, loss_val, it = network.train(dataset, I_train, R_train, val_set, I_val, R_val,\
                                             beta0_train, beta0_val, \
                                             training_steps, display_weights, save_path = args.save_model, saving_step = saving_step,  validate=True)

if not no_SIR:
    if not learn_alpha:
        b_train_nn, I_train_nn = network.compute_beta_I(dataset, I_train[:, 0], beta0_train)
        b_val_nn, I_val_nn = network.compute_beta_I(val_set, I_val[:, 0], beta0_val)
        if args.test_case:
            b_test_nn, I_test_nn = network.compute_beta_I(test_set, I_test[:, 0], beta0_test)
    else:
        b_train_nn, a_train_nn, I_train_nn, R_train_nn = network.compute_beta_I_alpha(dataset, I_train[:, 0], R_train[:,0], beta0_train, alpha)
        b_val_nn, a_val_nn,  I_val_nn, R_val_nn = network.compute_beta_I_alpha(val_set, I_val[:, 0], R_val[:,0], beta0_val, alpha)
        # if args.test_case:
        #     b_test_nn, I_test_nn = network.compute_beta_I_alpha(test_set, I_test[:, 0], R_test[:,0], beta0_test, alpha0_test)
else:
#no_SIR
    I_train_nn = network.compute_I(dataset, I_train[:, 0])
    I_val_nn = network.compute_I(val_set, I_val[:, 0])
    if args.test_case:
        I_test_nn = network.compute_I(test_set, I_test[:, 0])

########################
######################### plot di beta e infetti


# se è un test case allora si conoscono anche i beta esatti che quindi vengono
# plottati
if args.plot_train and args.test_case:
    plot_solutions_newloss.plot_beta_I(I_train_nn, b_train_nn, I_train, beta_train, \
                               "train", 5)
if args.plot_test and args.test_case:
    plot_solutions_newloss.plot_beta_I(I_test_nn, b_test_nn, I_test, beta_test, \
                               "test", 1)

# se si utilizzano dati reali i beta esatti sono sconosciuti, quindi vengono
# fatti i plot solo degli infetti reali, degli infetti calcolati dalla rete, e
# dei beta calcolati dalla rete
if not no_SIR:
    if not learn_alpha:
        if args.plot_train and not args.test_case:
            plot_solutions_newloss.plot_beta_I(I_train_nn, b_train_nn, I_train, n_iter = model.n_iter, set_type='train', plot_display=1,
                                       save_plots=args.save_plots)
        if args.plot_test and not args.test_case:
            plot_solutions_newloss.plot_beta_I(I_val_nn, b_val_nn, I_val, n_iter = model.n_iter, set_type='val', plot_display=1, save_plots=args.save_plots)
    else:
        if args.plot_train and not args.test_case:
            plot_solutions_newloss.plot_beta_I_alpha(I_train_nn, R_train_nn, b_train_nn, a_train_nn, I_train, R_train, n_iter = model.n_iter, set_type='train',
                                       save_plots=args.save_plots, plot = plot)
        if args.plot_test and not args.test_case:
            plot_solutions_newloss.plot_beta_I_alpha(I_val_nn, R_val_nn, b_val_nn, a_val_nn, I_val, R_val, n_iter = model.n_iter, set_type='val', save_plots=args.save_plots, plot = plot)
else:
#no_SIR
    if args.plot_train and not args.test_case:
        plot_solutions_newloss.plot_I(I_train_nn, I_train, n_iter = model.n_iter, set_type='train', plot_display=1,
                                   save_plots=args.save_plots)
    if args.plot_test and not args.test_case:
        plot_solutions_newloss.plot_I(I_val_nn, I_val, n_iter = model.n_iter, set_type='val', plot_display=1,
                                      save_plots=args.save_plots)

#################
# plot della loss
#################

if len(args.save_model) > 0:
    model.save_model(args.save_model)
if args.train:
    plt.plot(display_step * np.arange(0, it), loss_train[0:it])
    plt.plot(display_step * np.arange(0, it), loss_val[0:it])
    plt.legend(["loss train", "loss val"])
    plt.title("loss evolution")
    plt.show()
    plt.semilogy(display_step * np.arange(0, it), loss_train[0:it])
    plt.semilogy(display_step * np.arange(0, it), loss_val[0:it])
    plt.legend(["loss train", "loss val"])
    plt.title("loss evolution (semilog scale)")
    if len(args.save_plots) > 0:
        print("Saving the loss plot in " + args.save_plots + "...\n")
        path = "./" + args.save_plots;
        if not os.path.exists(path):
            os.mkdir(path)
        filepath2 = path + "/loss_plot.png";
        plt.savefig(fname=filepath2)
        plt.close()
        #plt.show()


