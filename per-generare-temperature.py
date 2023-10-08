# -*- coding: utf-8 -*-
"""
Created on Fri Jun 30 10:30:05 2023

@author: alean
"""
import math
import random
import MyTemperatureGenerator as tg
import MyDatasetGenerator as dsg
import numpy as np
import pickle
def T_base(t):
    return (1.2+math.cos(2*math.pi*(7/12-t)))/2.2

########################
# PARAMETRI DA SCEGLIERE
########################
N = 220
K = 17
K_test = 12
K_val = 6
train_fun_type = 'mixed'
val_fun_type = 'mixed'
test_fun_type = 'boy'

t = np.linspace(0, 1, N)
t_max = 1.0



def f(beta, T): # è la funzione che regola beta:   beta(t)' = f(beta(t), T(t))
    return 5.0*((1.0-T) - beta)

data = {  # questo dict viene usato per generare il dataset
    'beta0' : np.array([0.05]),
    'f' : f,
    't_max' : t_max,
    'N' : N
        }

temperature = []

for k in range(K):
    T_new = tg.generate_temperature(train_fun_type)
    T1_plot = np.zeros(N)
    for i in range(N):
        T1_plot[i] =  T_new(t[i])
    temperature.append(T_new)
    T2_plot = np.zeros(N)
    T = temperature[k]
    for i in range(N):
        T2_plot[i] =  T(t[i])
    del T_new
    

dataset = dsg.generate_dataset(temperature, data)   

temperature = []

for k in range(K_val):
    T_new = tg.generate_temperature(val_fun_type)
    T1_plot = np.zeros(N)
    for i in range(N):
        T1_plot[i] =  T_new(t[i])
    temperature.append(T_new)
    T2_plot = np.zeros(N)
    T = temperature[k]
    for i in range(N):
        T2_plot[i] =  T(t[i])
    del T_new
    

val_set = dsg.generate_dataset(temperature, data) 

temperature = []  

for k in range(K_test):
    T_new = tg.generate_temperature(test_fun_type)
    T1_plot = np.zeros(N)
    for i in range(N):
        T1_plot[i] =  T_new(t[i])
    temperature.append(T_new)
    T2_plot = np.zeros(N)
    T = temperature[k]
    for i in range(N):
        T2_plot[i] =  T(t[i])
    del T_new
    

test_set = dsg.generate_dataset(temperature, data)   

nome_file = 'LOAD_TEMP_N_'+str(N)+'_K_'+str(K)
if train_fun_type == 'mixed':
    nome_file=nome_file+'_MIXED.pkl'
else:
    nome_file=nome_file+'.pkl'


with open(nome_file, 'wb') as file:
    pickle.dump((dataset, K, val_set, K_val, test_set, K_test), file)
    

 
