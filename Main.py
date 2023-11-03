# -*- coding: utf-8 -*-
"""
Created on Tue May 23 12:45:34 2023

@author: alean
"""

"""
QUETA E' UNA PROVA PER IMPLEMENARE IL SIR
"""
import tensorflow as tf
import os
import numpy as np
import MyTemperatureGenerator as tg
import MyDatasetGenerator as dsg
from utilities import HerRungeKutta as hrk
import argparse
import sys
import pickle
from matplotlib import pyplot as plt
from utilities import SirHelperFunctions as shf
import random
import math

tfk = tf.keras
tfkl = tf.keras.layers
tfkl.Normalization(dtype='float64')

tf.keras.backend.set_floatx('float64')
seed = 100
random.seed(seed)
os.environ['PYTHONHASHSEED'] = str(seed)
np.random.seed(seed)
tf.random.set_seed(seed)
tf.compat.v1.set_random_seed(seed)

########################
# definizione dei flag
########################

parser = argparse.ArgumentParser()
parser.add_argument('-t', '--train', help='train the network', action='store_true')
parser.add_argument('-i', '--iterations', help='override the number of iterations', type=int, default=0)
parser.add_argument('-ph', '--print-help', action='store_true')
parser.add_argument('-s', '--save-weights', help='save the weights in the specified file after the training',default='')
parser.add_argument('-l', '--load-weights', help='load the weights in the specified file before the training',default='')
parser.add_argument('-lm', '--load-model', help='load the model in the specified file before the training',default='')
parser.add_argument('-f', '--file', help='specify the file name. default: data.txt', default='data.txt')
parser.add_argument('-p', '--plot', help='number of plots after training', default=0, type=int)
parser.add_argument('-n', '--new-weights', help='generate new random weights', action='store_true')
parser.add_argument('-nm', '--new-model', help='generate new model', action='store_true')
parser.add_argument('-ft', '--fun-type', help='override the type of temperature functions', default='')
parser.add_argument('-v', '--validate', help='use a validation set for the training', action='store_true')
parser.add_argument('-rp', '--random-plot', help='plot using randomly generated data', action='store_true')
parser.add_argument('-ptr', '--plot-train', help='plot using the training set', action='store_true')
parser.add_argument('-pte', '--plot-test', help='plot using the test set', action='store_true')
parser.add_argument('-lt', '--load-temp', help='load the temperatures', action='store_true')
parser.add_argument('-d', '--default', help='activate flags: --train, --validate, --load-temp', action='store_true')
parser.add_argument('-o', '--overwrite', help='save and load the weights from the specified file', default='')
parser.add_argument('-sp', '--save-plots', help='save the plots in the specified file after the training', default='')
parser.add_argument('-sm', '--save-model', help='save the model file in the specified directory after the training', default='')

args = parser.parse_args()

#######################
# stringa da inserire:
# 1) python prova-sir-keras.py --load-temp --train --save-model saved-models/nomemodello --plot-train --save-plots saved-plots/nomemodello
# o
# 2) python prova-sir-keras.py --load-temp --load-model saved-models/nomemodello --train --validate --save-model saved-models/nomemodello --plot-train --save-plots saved-plots/nomemodello
#python prova-sir-keras-2.py --load-temp --train --save-model saved-models/provaclassi --plot-train
#######################

if args.default:
    args.train = True
    args.validate = True
    args.load_temp = True

if len(args.overwrite) > 0:
    args.save_weights = args.overwrite
    args.load_weights = args.overwrite

# print-help da completare
if args.print_help:
    print('scrivere help per lo script')
    sys.exit()

###############################
# tutti i dati vengono presi dal file
###############################

from Data import *

####################################
# viene caricato il dataset
####################################

K_test = 0  # solo una variabile utile per i plot finali

if args.load_temp:
    nome_file_temp = 'datasets/'+ dataset
    try:
        with open(nome_file_temp, 'rb') as file:
            dataset, K, val_set, K_val, test_set, K_test = pickle.load(file)  # viene caricato il  dataset
        print('dataset', nome_file_temp, 'loaded...\n')
    except FileNotFoundError:
        print('file',nome_file_temp,'not found...\n')
        sys.exit()
#print(dataset[0,5,])
##############
# Viene creato o caricato il modello
##############

import ModelClass

# non c'è più bisogno di mettere new-model negli args, ma al massimo --load-model load_path
Model = ModelClass.Model(n_hidden = n_hidden, load_path = args.load_model, save_path = args.save_model )

##############

# vengono generati le osservazioni degli infetti

import InfectsGenerator as ig

I = ig.generate_I(dataset)
I_val = ig.generate_I(val_set)
# I = np.zeros([K, N])
# I_val = np.zeros([K_val, N])
# sir_0 = np.array([S0, I0, R0])
# for k in range(K):
#     s, i, r = hrk.RungeKutta(sir_0, dataset[1, k, ], N, t_max, alpha)
#     I[k, ] = i.copy()
# for k in range(K_val):
#     s, i, r = hrk.RungeKutta(sir_0, val_set[1, k, ], N, t_max, alpha)
#     I_val[k, ] = i.copy()
print(I.shape)
print((I[0]))
#####################
# training della rete
#####################

# #non c'è bisogno di inserire validate, viene fatto automaticamente
# if args.train:
#     loss_history, loss_history_val = Model.train(dataset, I, val_set, I_val)
#
# ########################
# # il modello viene salvato
# ########################
#
# if len(args.save_model) > 0:
#     print("Saving the model in " + args.save_model + "...\n")
#     Model.save_model(args.save_model)
#
# ######################
# # vengono fatti i plot
# ######################
#
# import plot_solutions as pt
# #plot a random
# if args.random_plot:
#     pt.test_plot(beta0, Model.g , ig.f, save_path = args.save_plots)
#
# #plot del training set
# if args.plot_train:
#     pt.Dataset_plot(dataset, beta0, Model.g, ig.f, train_or_test = 1, save_path = args.save_plots)
#     # plot della loss
#     pt.loss_plot(loss_history, loss_history_val, save_path=args.save_plots)
#
# #plot del test set
# if args.plot_test:
#     pt.Dataset_plot(dataset, beta0, Model.g, ig.f, train_or_test = 0, save_path = args.save_plots)
#









