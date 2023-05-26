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
import math
import MyTemperatureGenerator as tg
import MyDatasetGenerator as dsg
import HerRungeKutta as hrk
import argparse
import sys
import pickle
parser=argparse.ArgumentParser()
parser.add_argument('-t', '--train', action = 'store_true', help = 'train the network')
parser.add_argument('-ph','--print-help', action = 'store_true')
parser.add_argument('-s', '--save-weights', help = 'save the weights in the specified file after the training', action = 'store_true')
parser.add_argument('-fs', '--file-save', help = 'where to save the weigths. default: saved_weights.pkl', default = 'saved_weights.pkl')
parser.add_argument('-l', '--load-weights', help = 'load the weights in the specified file before the training', action = 'store_true')
parser.add_argument('-fl', '--file-load', help='where to save the weigths. default: saved_weights.pkl', default = 'saved_weights.pkl')
parser.add_argument('-f', '--file', help = 'specify the file name. default: data.txt')
parser.add_argument('-p', '--plot', help = 'number of plots after training', default = '0')

args = parser.parse_args()

if args.file_save:
    print(args.file_save)
    sys.exit()

if args.print_help:
    print('scrivere help per lo script')
    sys.exit()

data_dict = {}

with open(args.file, 'r') as file:
    for line in file:
        line = line.strip()  # Remove leading/trailing whitespace
        if line:
            field, value = line.split(':')
            data_dict[field.strip()] = value.strip()


learning_rate = float(data_dict['learning_rate'])
n_input = 2
n_hidden = 10
n_output = 1
display_step = 10


if args.load_weights:
    with open('saved_variables.pkl', 'rb') as file:
        weights, biases = pickle.load(args.file_load)

if not args.load_weights:  
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

t_max = float(data_dict['t_max'])
y0 = float(data_dict['beta0'])
N = int(data_dict['N'])
t = np.linspace(0, t_max, N)
dt = t_max/N
K=int(data_dict['K']) # numero di temperature
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
a=float(data_dict['alpha'])
sir_0= np.array([10, 0.5, 0.])
for k in range(K):
    s, i, r = hrk.RungeKutta(sir_0, dataset[1, k, ], N, t_max, a)
    I[k, ] = i.copy()
dt = t_max/N
#dataset = save_dataset
step_summation = int(data_dict['step_summation'])
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
    
training_steps = int(data_dict['training_steps'])
display_step = int(data_dict['display_step'])

if args.train:
    for i in range(training_steps):
      train_step()
      if i % display_step == 0:
        print("iterazione %i:" % i)
        print("loss: %f " % (custom_loss()))
        

if args.save_weights:
    with open(args.file_save, 'wb') as file:
        pickle.dump((weights, biases), args.file_save)

n_plots = int(args.plot)
for i in range(n_plots):
    runfile('plot-sir.py')


