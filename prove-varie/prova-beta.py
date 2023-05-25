# -*- coding: utf-8 -*-
"""
Created on Thu May 11 10:20:06 2023

@author: alean
"""

import numpy as np
import tensorflow as tf
import math
import MyCrankNicolsonClass as cnc
from matplotlib import pyplot as plt
import os
import random
import MyDatasetGenerator as dg

tf.keras.backend.set_floatx('float64')
tfk = tf.keras
tfkl = tf.keras.layers

# Random seed for reproducibility 
#seed = 42
seed = 64
random.seed(seed)
os.environ['PYTHONHASHSEED'] = str(seed)
np.random.seed(seed)
tf.random.set_seed(seed)
tf.compat.v1.set_random_seed(seed)

learning_rate = 0.01
training_steps = 250
batch_size = 100
display_step = 25
# Network Parameters
n_input = 2     # input layer number of neurons
n_hidden_1 = 2 # 1st layer number of neurons
n_hidden_2 = 2 # 2nd layer number of neurons
n_output = 1    # output layer number of neurons

initializer = tf.keras.initializers.GlorotNormal()

weights = {
'h1': tf.Variable(initializer(shape = (n_input, n_hidden_1))),
'h2': tf.Variable(initializer(shape = (n_hidden_1, n_hidden_2))),
'out': tf.Variable(initializer(shape = (n_hidden_2, n_output)))
}
biases = {
'b1': tf.Variable(initializer(shape = (n_hidden_1,))),
'b2': tf.Variable(initializer(shape = (n_hidden_2,))),
'out': tf.Variable(initializer(shape = (n_output,)))
}
# Stochastic gradient descent optimizer.
optimizer = tf.optimizers.SGD(learning_rate)

# Create model
def multilayer_perceptron(x):
  # x = np.array([[[x]]],  dtype='float32')
  layer_1 = tf.add(tf.matmul(x, weights['h1']), biases['b1'])
  layer_1 = tf.nn.leaky_relu(layer_1)
  #layer_1 = tf.nn.sigmoid(layer_1)
  layer_2 = tf.add(tf.matmul(layer_1, weights['h2']), biases['b2'])
  layer_2 = tf.nn.leaky_relu(layer_2)
  #layer_2 = tf.nn.sigmoid(layer_2)
  output = tf.matmul(layer_2, weights['out']) + biases['out']
  return output
# Universal Approximator
def g(beta, T):
    # x = np.array([beta, T])
    TT = tf.constant([[T]],dtype = 'float64')
    x = tf.concat([beta, TT], 1)
    #return x * multilayer_perceptron(x)
    return multilayer_perceptron(x)
# Given EDO
tau =10.0

def beta_eq(T):
    return 16.0 - T
"""
def T(t):
    return -t**2+5*t
"""

# QUI VENGONO GENERATE DELLE TEMPERATURE
temperature_funcs = []
t_max=1.0
K = 10 # numero di funzioni temperatura
for k in range(K):
    center = random.gauss(0.5, 0.1)
    height = random.gauss(16.0, 2.0)
    width = random.gauss(14.0, 1.0)
    def T(t):
        return math.sin(2*math.pi/t_max*(t-center))*width + height
    temperature_funcs.append(T)
     

# QUI VIENE DEFINITO IL SISTEMA: dbeta = f(beta(t), T(t))

N = 50
beta0 = np.array([0.5])
def f(beta, T):
    return (beta_eq(T) - beta[0])/tau

data = {
    "N" : N,
    "t_max" : t_max,
    "beta0" : beta0,
    "f" : f
}

dataset = dg.generate_dataset(temperature_funcs, data)



def dbeta_hat(beta, t):
    return g(beta[0], T(t))
    
# Custom loss function to approximate the derivatives

"""
cn_solver = cnc.CrankNicolson(sys, beta0, t_max, N)
cn_solver.compute_solution()
t, beta = cn_solver.get_solution()
dt = t_max /N
"""
"""
def custom_loss():
    
    curr_beta = beta0[0]
    next_beta = curr_beta
    summation = []
    for i in range(beta.shape[1]-1):
        # x = np.array([curr_beta, T(i*dt)])
        # next_beta = curr_beta + dt * multilayer_perceptron( x )
        next_beta = curr_beta + dt * g(curr_beta, T(i*dt))
        summation.append( dt*(beta[0, i+1] - next_beta)**2 )
        curr_beta = next_beta.numpy()[0][0][0][0]
    return tf.sqrt(tf.reduce_sum(tf.abs(summation)))
"""

# UNA NUOVA CUSTOM_LOSS CHE UTILIZZA TUTTE LE TEMPERATURE
dt = t_max / N
summation_step = 5
def custom_loss():
    summation = []
    for k in range(K):
        curr_beta = tf.constant([[beta0[0]]], dtype = 'float64')
        next_beta = curr_beta
        dt_tensor = tf.constant([[dt]], dtype = 'float64')
        for i in range(N-1):
            #next_beta = curr_beta + dt * g(curr_beta, dataset[0, k, i])
            next_beta = tf.add(curr_beta, tf.matmul(dt_tensor, g(curr_beta, dataset[0, k, i])) )
            if i % summation_step == 0:
                real_beta=tf.constant([[dataset[1,k,i+1]]], dtype = 'float64')
                summation.append((real_beta - next_beta)**2)
            curr_beta = next_beta
    return (tf.reduce_mean(tf.abs(summation)))

def train_step():
    with tf.GradientTape() as tape:
        loss = custom_loss()
    trainable_variables=list(weights.values())+list(biases.values())
    gradients = tape.gradient(loss, trainable_variables)
    optimizer.apply_gradients(zip(gradients, trainable_variables))
# Training the Model:
for i in range(training_steps):
  train_step()
  if i % display_step == 0:
    print("iterazione numero: %i " %(i))
    print("loss: %f " % (custom_loss()))

"""
curr_beta = beta0[0]
beta_hat = np.zeros(beta.shape[1])
beta_hat[0]=curr_beta
for i in range(beta.shape[1]-1):
    # x = np.array([curr_beta, T(i*dt)])
    # print(curr_beta)
    next_beta = curr_beta + dt * g( curr_beta, T(dt*i) ).numpy()[0][0][0][0]
    beta_hat[i+1] = next_beta
    curr_beta = next_beta
"""

# ORA GENERO UNA NUOVA TEMPERATURA DIFFERENTE DALLE PRECEDENTI E FACCIO UN PLOT

center = random.gauss(0.5, 0.1)
height = random.gauss(16.0, 2.0)
width = random.gauss(14.0, 1.0)
def T(t):
    return math.sin(2*math.pi/tau*(t-center))*width + height



def dbeta(beta, t):
    return f(beta, T(t))
sys = [dbeta]
cn_solver = cnc.CrankNicolson(sys, beta0, t_max, N)
cn_solver.compute_solution()
t, beta = cn_solver.get_solution()
curr_beta = tf.constant([[beta0[0]]])
beta_hat = np.zeros(beta.shape[1])
beta_hat[0]=curr_beta.numpy()[0][0]
for i in range(beta.shape[1]-1):
    next_beta = curr_beta + dt * g( curr_beta, T(dt*i) ).numpy()[0][0]
    beta_hat[i+1] = next_beta.numpy()[0][0]
    curr_beta = next_beta

pp = plt.plot(t, beta_hat)
plt.plot(t, beta[0, ])
plt.legend(["soluzione rete", "solzione vera"])







