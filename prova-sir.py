# -*- coding: utf-8 -*-
"""
Created on Tue May 23 12:45:34 2023

@author: alean
"""

"""
QUETA E' UNA PROVA PER IMPLEMENARE IL SIR
"""
import tensorflow as tf
import numpy as np
from matplotlib import pyplot as plt
import math
import MyTemperatureGenerator as tg
import MyCrankNicolsonClass as cnc
import MyDatasetGenerator as dsg
import HerRungeKutta as hrk

learning_rate = 0.01
n_input = 2
n_hidden = 32
n_output = 1
display_step = 10


weights = {
'h1': tf.Variable(tf.random.normal([n_input, n_hidden])),
'out': tf.Variable(tf.random.normal([n_hidden, n_output]))
}
biases = {
'b1': tf.Variable(tf.random.normal([n_hidden])),
'out': tf.Variable(tf.random.normal([n_output]))
}


# Stochastic gradient descent optimizer.
optimizer = tf.optimizers.SGD(learning_rate)



# Create model
def multilayer_perceptron(x):
  layer_1 = tf.add(tf.matmul(x, weights['h1']), biases['b1'])
  layer_1 = tf.nn.sigmoid(layer_1)
  output = tf.matmul(layer_1, weights['out']) + biases['out']
  return output

    
def g(y, v):
    tv = tf.constant([[v]], dtype = 'float32')
    x = tf.concat([y, tv], 1)
    return multilayer_perceptron(x)

t_max = 1.0
y0=0.05
N = 60
t = np.linspace(0, t_max, N)
dt = t_max/N
K=10 # numero di temperature
temperature= []
I = np.zeros([K, N]) #da usare per gli infetti
def f(beta, T):
    return 5.0*((1.0-T) - beta)
for k in range(K):
    T = tg.generate_temperature("exp", t_max=t_max)
    temperature.append(T)

data = {
    'beta0' : np.array([y0]),
    'f' : f,
    't_max' : t_max,
    'N' : N
        }

#save_dataset = dataset.copy()
dataset = dsg.generate_dataset(temperature, data)
a=3.0
sir_0= np.array([10, 0.5, 0.])
for k in range(K):
    s, i, r = hrk.RungeKutta(sir_0, dataset[1, k, ], N, t_max, a)
    I[k, ] = i.copy()
dt = t_max/N
#dataset = save_dataset
step_summation = 1
def custom_loss():
    total_summation = []
    for k in range(K):
        curr_y = tf.constant([[y0]], dtype = 'float32')
        summation = []
        curr_I_nn = tf.constant([[sir_0[1]]], dtype = 'float32')
        curr_S_nn = tf.constant([[sir_0[0]]], dtype= 'float32')
        for i in range(N-1):
            next_y = curr_y + dt*g(curr_y, dataset[0, k, i])
            next_S_nn = curr_S_nn - dt*tf.matmul(curr_y, tf.matmul(curr_S_nn, curr_I_nn))
            next_I_nn = curr_I_nn + dt*(tf.matmul(curr_y, tf.matmul(curr_S_nn, curr_I_nn))) - dt*a*curr_I_nn
            #print(next_I_nn)
            #print(I[k, i+1])
            if i % step_summation == 0:
                summation.append((next_I_nn - I[k, i+1])**2)
            curr_y = next_y
            curr_S_nn = next_S_nn
            curr_I_nn = next_I_nn
        total_summation.append(tf.reduce_sum(summation))
        #print(tf.reduce_sum(total_summation))
    return tf.reduce_mean(total_summation)
    

def train_step():
    with tf.GradientTape() as tape:
        loss = custom_loss()
    trainable_variables=list(weights.values())+list(biases.values())
    gradients = tape.gradient(loss, trainable_variables)
    optimizer.apply_gradients(zip(gradients, trainable_variables))
    
training_steps = 500
display_step = 50

for i in range(training_steps):
  train_step()
  if i % display_step == 0:
    print("iterazione %i:" % i)
    print("loss: %f " % (custom_loss()))
