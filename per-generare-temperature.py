# -*- coding: utf-8 -*-
"""
Created on Fri Jun 30 10:30:05 2023

@author: alean
"""
import math
import random
import MyTemperatureGenerator as tg
# import MyDatasetGenerator as dsg
import numpy as np
import pickle
def T_base(t):
    return (1.2+math.cos(2*math.pi*(7/12-t)))/2.2

########################
# PARAMETRI DA SCEGLIERE
########################
data_dict = {}

with open('data.txt', 'r') as file:
    for line in file:
        line = line.strip()  # Remove leading/trailing whitespace
        if line:
            field, value = line.split(':')
            data_dict[field.strip()] = value.strip()

N = 150
K = 40
K_test = 20
K_val = 40
train_fun_type = 'adriano-style'
val_fun_type = 'adriano-style'
test_fun_type = 'adriano-style'
alpha = 2.0
S0 = 0.99
S_inf = 0.03
b_ref = alpha*math.log(S0/S_inf)/(1-S_inf)
nome_file = "prova_con_beta0.pkl"
beta0_inf = 0.5
beta0_sup = 5.0



t = np.linspace(0, 1, N)
t_max = 12.0

"""
def f(beta, betaeq): # è la funzione che regola beta:   beta(t)' = f(beta(t), T(t))
    return (1/tau)*(betaeq - beta)*t_max

data = {  # questo dict viene usato per generare il dataset
    'beta0' : np.array([0.5]),
    'f' : f,
    't_max' : 1.0,
    'N' : N
        }

temperatureandbetaeqs = []

for k in range(K):
    if train_fun_type == 'adriano-style':
        T_new, Betaeqnew = tg.generate_temp_by_adri(b_ref)
    else:
        T_new = tg.generate_temperature(train_fun_type)
        Betaeqnew = []
    # T1_plot = np.zeros(N)
    # for i in range(N):
    #     T1_plot[i] =  T_new(t[i])
    temperatureandbetaeqs.append((T_new,Betaeqnew))
    # T2_plot = np.zeros(N)
    #T = temperature[k]
    # for i in range(N):
    #     T2_plot[i] =  T(t[i])
    # del T_new
    

dataset = dsg.generate_dataset(temperatureandbetaeqs, data)

temperatureandbetaeqs = []

for k in range(K_val):
    if train_fun_type == 'adriano-style':
        T_new, Betaeqnew = tg.generate_temp_by_adri(b_ref)
    else:
        T_new = tg.generate_temperature(val_fun_type)
        Betaeqnew = []
    # T1_plot = np.zeros(N)
    # for i in range(N):
    #     T1_plot[i] =  T_new(t[i])
    temperatureandbetaeqs.append((T_new, Betaeqnew))
    # T2_plot = np.zeros(N)
    # T = temperature[k]
    # for i in range(N):
    #     T2_plot[i] =  T(t[i])
    # del T_new
    

val_set = dsg.generate_dataset(temperatureandbetaeqs, data)

temperatureandbetaeqs = []

for k in range(K_test):
    if train_fun_type == 'adriano-style':
        T_new, Betaeqnew = tg.generate_temp_by_adri(b_ref)
    else:
        T_new = tg.generate_temperature(test_fun_type)
        Betaeqnew = []
    # T1_plot = np.zeros(N)
    # for i in range(N):
    #     T1_plot[i] =  T_new(t[i])
    temperatureandbetaeqs.append((T_new, Betaeqnew))
    # T2_plot = np.zeros(N)
    # T = temperature[k]
    # for i in range(N):
    #     T2_plot[i] =  T(t[i])
    # del T_new
    

test_set = dsg.generate_dataset(temperatureandbetaeqs, data)
if train_fun_type == 'adriano-style':
    nome_file = 'ADRIANO_STYLE_TEMP_N_'+str(N)+'_K_'+str(K)
else:
    nome_file = 'LOAD_TEMP_N_'+str(N)+'_K_'+str(K)

nome_file = nome_file + '_mod_'
if train_fun_type == 'mixed':
    nome_file=nome_file+'_MIXED.pkl'
else:
    nome_file=nome_file+'.pkl'

"""
dataset = np.zeros([K, N])
beta0_train = np.random.uniform(beta0_inf, beta0_sup, (K))
for k in range(K):
    if train_fun_type == 'adriano-style':
        T_new, Betaeqnew = tg.generate_temp_by_adri(b_ref)
    else:
        T_new = tg.generate_temperature(val_fun_type)
    for i in range(N):
        dataset[k, i] = T_new(t[i])
    del T_new

beta0_val = np.random.uniform(beta0_inf, beta0_sup, (K_val))
val_set = np.zeros([K_val, N])
for k in range(K_val):
    if train_fun_type == 'adriano-style':
        T_new, Betaeqnew = tg.generate_temp_by_adri(b_ref)
    else:
        T_new = tg.generate_temperature(val_fun_type)
    for i in range(N):
        val_set[k, i] = T_new(t[i])
    del T_new

beta0_test = np.random.uniform(beta0_inf, beta0_sup, (K_test))
test_set = np.zeros([K_test, N])
for k in range(K_test):
    if train_fun_type == 'adriano-style':
        T_new, Betaeqnew = tg.generate_temp_by_adri(b_ref)
    else:
        T_new = tg.generate_temperature(val_fun_type)
    for i in range(N):
        test_set[k, i] = T_new(t[i])
    del T_new

with open('datasets/'+nome_file, 'wb') as file:
    pickle.dump((dataset, val_set, test_set, beta0_train, beta0_val, beta0_test), file)
    

 
