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
import MyTemperatureGenerator as tg
import MyDatasetGenerator as dsg
import HerRungeKutta as hrk
import argparse
import sys
import pickle
from matplotlib import pyplot as plt
parser=argparse.ArgumentParser()
parser.add_argument('-t', '--train', help = 'train the network', action = 'store_true')
parser.add_argument('-i', '--iterations', help = 'override the number of iterations', type = int, default = 0)
parser.add_argument('-ph','--print-help', action = 'store_true')
parser.add_argument('-s', '--save-weights', help = 'save the weights in the specified file after the training', default = '')
# parser.add_argument('-fs', '--file-save', help = 'where to save the weigths. default: saved_weights.pkl', default = 'saved_weights.pkl')
parser.add_argument('-l', '--load-weights', help = 'load the weights in the specified file before the training', default = '')
# parser.add_argument('-fl', '--file-load', help='where to save the weigths. default: saved_weights.pkl', default = 'saved_weights.pkl')
parser.add_argument('-f', '--file', help = 'specify the file name. default: data.txt', default = 'data.txt')
parser.add_argument('-p', '--plot', help = 'number of plots after training', default = 0, type = int)
parser.add_argument('-n', '--new-weights', help='generate new random weights', action='store_true')
parser.add_argument('-ft', '--fun-type', help = 'override the type of temperature functions', default = '')

args = parser.parse_args()


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
n_hidden = 15
n_output = 1
display_step = 10


if len(args.load_weights) > 0:
    with open(args.load_weights, 'rb') as file:
        weights, biases = pickle.load(file)

if args.new_weights:  
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
    if len(args.fun_type) == 0:
        T = tg.generate_temperature(data_dict["temperature_type"], t_max=t_max)
    else:
        T = tg.generate_temperature(args.fun_type, t_max=t_max)
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
S0 = float(data_dict['S0'])
I0 = float(data_dict['I0'])
R0 = float(data_dict['R0'])
sir_0= np.array([S0, I0, R0])
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
if args.iterations == 0:
    training_steps = int(data_dict['training_steps'])
else:
    training_steps = args.iterations
display_step = int(data_dict['display_step'])

if args.train:
    try:
        for i in range(training_steps):
          train_step()
          if i % display_step == 0:
            print("iterazione %i:" % i)
            print("loss: %f " % (custom_loss()))
    except KeyboardInterrupt:
        print('Training interrupted by user. Proceeding to save the weights and plot the solutions')
        

if len(args.save_weights) > 0:
    with open(args.save_weights, 'wb') as file:
        pickle.dump((weights, biases), file)

n_plots = args.plot

for i in range(n_plots):
    plt.show()
    if len(args.fun_type) == 0:
        T = tg.generate_temperature(data_dict["temperature_type"], t_max = t_max)
    else:
        T = tg.generate_temperature(args.fun_type, t_max = t_max)
    y_real = np.zeros(N)
    y_nn = np.zeros(N)
    y_real[0] = y0
    y_nn[0] = y0 
    curr_y = tf.constant([[y0]], dtype = 'float32')
    for i in range(N-1):
        next_y = curr_y + dt*g(curr_y, T(t[i]))
        y_nn[i+1] = next_y.numpy()[0][0]
        y_real[i+1] = y_real[i] + dt*f(y_real[i], T(t[i]))
        curr_y = next_y
    plt.plot(t, y_real)
    plt.plot(t, y_nn)
    plt.legend(["soluzione reale", "soluzione rete"])

