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
tf.keras.backend.set_floatx('float32')

f0 = 1
inf_s = np.sqrt(np.finfo(np.float32).eps)
learning_rate = 0.5
training_steps = 200
batch_size = 100
display_step = 20
# Network Parameters
n_input = 1     # input layer number of neurons
n_hidden_1 = 2 # 1st layer number of neurons
n_hidden_2 = 2 # 2nd layer number of neurons
n_output = 1    # output layer number of neurons
weights = {
'h1': tf.Variable(tf.random.normal([n_input, n_hidden_1])),
'h2': tf.Variable(tf.random.normal([n_hidden_1, n_hidden_2])),
'out': tf.Variable(tf.random.normal([n_hidden_2, n_output]))
}
biases = {
'b1': tf.Variable(tf.random.normal([n_hidden_1])),
'b2': tf.Variable(tf.random.normal([n_hidden_2])),
'out': tf.Variable(tf.random.normal([n_output]))
}
# Stochastic gradient descent optimizer.
optimizer = tf.optimizers.SGD(learning_rate)

# Create model
def multilayer_perceptron(x):
  # x = np.array([[[x]]],  dtype='float32')
  layer_1 = tf.add(tf.matmul(x, weights['h1']), biases['b1'])
  layer_1 = tf.nn.sigmoid(layer_1)
  layer_2 = tf.add(tf.matmul(layer_1, weights['h2']), biases['b2'])
  layer_2 = tf.nn.sigmoid(layer_2)
  output = tf.matmul(layer_2, weights['out']) + biases['out']
  return output
# Universal Approximator
def g(beta):
    # x = np.array([beta, T])
    #return x * multilayer_perceptron(x)
    return multilayer_perceptron(beta)
# Given EDO
tau = 365.

def beta_eq(T):
    return 25.0 - T

def T(t):
    return math.cos(t)

def dbeta(beta, t):
    return  -6.0*t*beta[0]
def dbeta_hat(beta, t):
    return g(beta[0], T(t))
    
sys = [dbeta]    

def f(x):
  return 2*x
# Custom loss function to approximate the derivatives
"""
def custom_loss():
  summation = []
  t_max=1.0
  N=50
  beta0=np.array([0.5])
  cn_solver = cnc.CrankNicolson(sys, beta0, t_max, N)
  cn_solver.compute_solution()
  t, beta = cn_solver.get_solution()
  sys_hat=[dbeta_hat]
  cn_solver_hat = cnc.CrankNicolson(sys_hat, beta0, t_max, N)
  cn_solver_hat.compute_solution()
  t, beta_hat = cn_solver_hat.get_solution()
  for i in range(len(beta)):
      summation.append( ( beta[i] - beta_hat[i] )**2 )
  return tf.sqrt(tf.reduce_mean(tf.abs(summation)))
"""

t_max=1.0
N=100
beta0=np.array([1.0])
cn_solver = cnc.CrankNicolson(sys, beta0, t_max, N)
cn_solver.compute_solution()
t, beta = cn_solver.get_solution()
dt = t_max /N


def custom_loss():
    curr_beta = tf.constant([[beta0[0]]], dtype = 'float32')
    #next_beta = curr_beta
    summation = []
    for i in range(beta.shape[1]-1):
        # x = np.array([curr_beta, T(i*dt)])
        next_beta = curr_beta + dt * g( curr_beta )
        #next_beta = curr_beta + dt * g(curr_beta)
        real_beta = tf.constant([[beta[0, i+1]]], dtype = 'float32')
        summation.append( (real_beta - next_beta)**2 )
        curr_beta = next_beta
    return tf.reduce_mean(summation)
"""
def f_prova(x):
    return x**2

def custom_loss2():
    x = np.linspace(-1.0, 1.0, 100)
    summation = []
    for i in range(len(x)):
        summation.append( ( f_prova(x[i]) - g(x[i]) )**2 )
    return tf.sqrt(tf.reduce_mean(tf.abs(summation)))
"""

def train_step():
    with tf.GradientTape() as tape:
        loss = custom_loss()
    trainable_variables=list(weights.values())+list(biases.values())
    gradients = tape.gradient(loss, trainable_variables)
    optimizer.apply_gradients(zip(gradients, trainable_variables))
    return gradients

gradients = train_step()

# Training the Model:
for i in range(training_steps):
  train_step()
  #print(gradients)
  if i % display_step == 0:
    print("loss: %f " % (custom_loss()))


curr_beta = tf.constant([[beta0[0]]], dtype = 'float32')
beta_hat = np.zeros(beta.shape[1])
beta_hat[0]=curr_beta
for i in range(beta.shape[1]-1):
    # x = np.array([curr_beta, T(i*dt)])
    # print(curr_beta)
    next_beta = curr_beta + dt * g( curr_beta )
    beta_hat[i+1] = next_beta.numpy()[0][0]
    curr_beta = next_beta
    
    
plt.plot(t, beta_hat)
cn_solver.plot_solutions()






