
"""
Created on Fri Jun 30 10:30:05 2023

@author: alean
"""
import math
import random
from utilities import MyTemperatureGenerator as tg
import numpy as np
import pickle
import argparse


data_dict = {}

with open('utilities/data-for-GenerateDataset.txt', 'r') as file:
    for line in file:
        line = line.strip()  # Remove leading/trailing whitespace
        if line:
            field, value = line.split(':')
            data_dict[field.strip()] = value.strip()

###############################
# VENGONO PRESI I DATI DAL FILE
###############################

N = int(data_dict['N'])
K = int(data_dict['K'])
K_val = int(data_dict['K_val'])
K_test = int(data_dict['K_test'])
train_fun_type = 'adriano-style'
val_fun_type = 'adriano-style'
test_fun_type = 'adriano-style'
S0 = float(data_dict['S0'])
S_inf = float(data_dict['S_inf'])
nome_file = data_dict['nome_file']
beta0_inf = float(data_dict['beta0_inf'])
beta0_sup = float(data_dict['beta0_sup'])
t_max = float(data_dict['t_max'])
alpha = float(data_dict['alpha'])

b_ref = alpha*math.log(S0/S_inf)/(1-S_inf)
t = np.linspace(0, 1, N)


dataset = np.zeros([K, N, 2])
beta0_train = np.random.uniform(beta0_inf, beta0_sup, (K))
for k in range(K):
    if train_fun_type == 'adriano-style':
        T_new, Betaeqnew = tg.generate_temp_by_adri(b_ref)
    else:
        T_new = tg.generate_temperature(val_fun_type)
    
    lockdown_function = tg.generate_lockdown_function()
    
    for i in range(N):
        dataset[k, i, 0] = T_new(t[i])
        dataset[k, i, 1] = lockdown_function(t[i])
    del T_new, lockdown_function

beta0_val = np.random.uniform(beta0_inf, beta0_sup, (K_val))
val_set = np.zeros([K_val, N, 2])
for k in range(K_val):
    if train_fun_type == 'adriano-style':
        T_new, Betaeqnew = tg.generate_temp_by_adri(b_ref)
    else:
        T_new = tg.generate_temperature(val_fun_type)
    
    lockdown_function = tg.generate_lockdown_function()
    
    for i in range(N):
        val_set[k, i, 0] = T_new(t[i])
        val_set[k, i, 1] = lockdown_function(t[i])
    del T_new, lockdown_function


beta0_test = np.random.uniform(beta0_inf, beta0_sup, (K_test))
test_set = np.zeros([K_test, N, 2])
for k in range(K_test):
    if train_fun_type == 'adriano-style':
        T_new, Betaeqnew = tg.generate_temp_by_adri(b_ref)
    else:
        T_new = tg.generate_temperature(val_fun_type)
    
    lockdown_function = tg.generate_lockdown_function()

    
    for i in range(N):
        test_set[k, i, 0] = T_new(t[i])
        test_set[k, i, 1] = lockdown_function(t[i])
    del T_new

with open('datasets/'+nome_file, 'wb') as file:
    pickle.dump((dataset, val_set, test_set, beta0_train, beta0_val, beta0_test), file)
    
"""
n_input = 2
n_hidden = 15
n_output = 1
weights = {
'h1': tf.Variable(tf.random.normal([n_input, n_hidden], dtype='float64'), dtype='float64'),
'out': tf.Variable(tf.random.normal([n_hidden, n_output], dtype='float64'), dtype='float64')
}
biases = {
'b1': tf.Variable(tf.random.normal([n_hidden], dtype='float64'), dtype='float64'),
'out': tf.Variable(tf.random.normal([n_output], dtype='float64'), dtype='float64')
}

def multilayer_perceptron(x):
    layer_1 = tf.add(tf.matmul(x, weights['h1']), biases['b1'])
    layer_1 = tf.nn.sigmoid(layer_1)
    output = tf.matmul(layer_1, weights['out']) +biases['out']
    return output

def g(y, v):
    v.shape = (v.shape[0], n_input-1)
    tv = tf.constant(v, dtype='float64')
    x = tf.concat([y, tv], 1)
    return multilayer_perceptron(x)

 
"""