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
tf.keras.backend.set_floatx('float64')

########################
# definizione dei flag
########################

parser=argparse.ArgumentParser()
parser.add_argument('-t', '--train', help = 'train the network', action = 'store_true')
parser.add_argument('-i', '--iterations', help = 'override the number of iterations', type = int, default = 0)
parser.add_argument('-ph','--print-help', action = 'store_true')
parser.add_argument('-s', '--save-weights', help = 'save the weights in the specified file after the training', default = '')
parser.add_argument('-l', '--load-weights', help = 'load the weights in the specified file before the training', default = '')
parser.add_argument('-f', '--file', help = 'specify the file name. default: data.txt', default = 'data.txt')
parser.add_argument('-p', '--plot', help = 'number of plots after training', default = 0, type = int)
parser.add_argument('-n', '--new-weights', help='generate new random weights', action='store_true')
parser.add_argument('-ft', '--fun-type', help = 'override the type of temperature functions', default = '')
parser.add_argument('-v', '--validate', help = 'use a validation set for the training', action = 'store_true')
parser.add_argument('-pt', '--plot-train', help='plot using the train set', action = 'store_true')
parser.add_argument('-lt', '--load-temp', help = 'load the temperatures', action = 'store_true')
parser.add_argument('-d', '--default',  help = 'activate flags: --train, --validate, --load-temp', action = 'store_true')
parser.add_argument('-o', '--overwrite', help = 'save and load the weights from the specified file', default = '')

args = parser.parse_args()

if args.default:
    args.train = True
    args.validate = True
    args.load_temp = True
    
if len(args.overwrite)>0:
    args.save_weights = args.overwrite
    args.load_weights = args.overwrite

if len(args.load_weights)>0 and args.new_weights:
    print("Cannot generate new weights and loading existing ones. Stopping execution..")
    sys.exit()

# print-help da completare

if args.print_help:
    print('scrivere help per lo script')
    sys.exit()

# i dati vengono presi dal file

data_dict = {}

with open(args.file, 'r') as file:
    for line in file:
        line = line.strip()  # Remove leading/trailing whitespace
        if line:
            field, value = line.split(':')
            data_dict[field.strip()] = value.strip()


learning_rate = float(data_dict['learning_rate'])
n_input = 2
n_hidden = int(data_dict['n_hidden']) # il numero di hidden neurons viene preso dal file data.txt
n_output = 1
display_step = int(data_dict['display_step'])

####################################
# vengono generati o caricati i pesi
####################################
K_test = 0 # solo una variabile utile per i plot finali
if len(args.load_weights) > 0:
    print("Loading weights from "+args.load_weights+"...\n")
    with open(args.load_weights, 'rb') as file:
        weights, biases, dataset = pickle.load(file)
if args.load_temp:
    print("Loading the datasets...\n")
    nome_file_temp = "LOAD_TEMP.pkl"
    with open(nome_file_temp, 'rb') as file:
        dataset, K, val_set, K_val, test_set, K_test = pickle.load(file)  # viene caricato il  dataset
        

if args.new_weights:  
    print("Generating new weights...\n")
    weights = {
    'h1': tf.Variable(tf.random.normal([n_input, n_hidden], dtype='float64'), dtype='float64'),
    'out': tf.Variable(tf.random.normal([n_hidden, n_output], dtype='float64'), dtype='float64')
    }
    biases = {
    'b1': tf.Variable(tf.random.normal([n_hidden], dtype='float64'), dtype='float64'),
    'out': tf.Variable(tf.random.normal([n_output], dtype='float64'), dtype='float64')
    }


# Stochastic gradient descent optimizer.
optimizer = tf.optimizers.SGD(learning_rate)
# optimizer = tf.optimizers.Adam(learning_rate)

##############
# Create model
##############

def multilayer_perceptron(x):
  layer_1 = tf.add(tf.matmul(x, weights['h1']), biases['b1'])
  layer_1 = tf.nn.sigmoid(layer_1)
  output = tf.matmul(layer_1, weights['out']) + biases['out']
  return output

    
def g(y, v):
    tv = tf.constant([[v]], dtype = 'float64')
    x = tf.concat([y, tv], 1)
    return multilayer_perceptron(x)

t_max = float(data_dict['t_max'])
y0 = float(data_dict['beta0'])
N = int(data_dict['N'])
t = np.linspace(0, t_max, N)
dt = t_max/N
if not args.load_temp: # se le temperature non sono state caricate, il num di temperature è letto da data.txt
    K=int(data_dict['K']) # numero di temperature
K_val = int(data_dict['K_val']) # numero di temperature da usare nel validation set
temperature= []
temperature_val = []


# vengono generate le temperature per il training

def f(beta, T): # è la funzione che regola beta:   beta(t)' = f(beta(t), T(t))
    return 5.0*((1.0-T) - beta)


data = {  # questo dict viene usato per generare il dataset
    'beta0' : np.array([y0]),
    'f' : f,
    't_max' : t_max,
    'N' : N
        }


if len(args.load_weights) == 0 and not args.load_temp: # se i pesi e le temp non sono stati caricati, vengono generate nuove temp
    for k in range(K):
        if len(args.fun_type) == 0: # override dei tipi di temperature
            T = tg.generate_temperature(data_dict["temperature_type"], t_max=t_max)
        else:
            T = tg.generate_temperature(args.fun_type, t_max=t_max)
        temperature.append(T)
    dataset = dsg.generate_dataset(temperature, data)


if args.validate and not args.load_temp: # vengono generate le temperature da usare nel validation set
    for k in range(K_val):
        if len(args.fun_type) == 0:
            T_val = tg.generate_temperature(data_dict["temperature_type"], t_max = t_max)
        else:
            T_val = tg.generate_temperature(args.fun_type, t_max = t_max)
        temperature_val.append(T_val)
    val_set = dsg.generate_dataset(temperature_val, data)

    

# vengono generati le osservazioni degli infetti

I = np.zeros([K, N]) #da usare per gli infetti
if args.validate:
    I_val = np.zeros([K_val, N]) # da usare per il validation set
a = float(data_dict['alpha'])
S0 = float(data_dict['S0'])
I0 = float(data_dict['I0'])
R0 = float(data_dict['R0'])
TOT = S0 + I0 + R0 # popolazione totale
sir_0 = np.array([S0, I0, R0])
for k in range(K):
    s, i, r = hrk.RungeKutta(sir_0, dataset[1, k, ], N, t_max, a)
    I[k, ] = i.copy()
if args.validate:
    for k in range(K_val):
        s, i, r = hrk.RungeKutta(sir_0, val_set[1, k, ], N, t_max, a)
        I_val[k, ] = i.copy()

dt = t_max/N
step_summation = int(data_dict['step_summation'])

########################
# definizione della loss
########################

def custom_loss(K, dataset):
    total_summation = []
    for k in range(K):
        curr_y = tf.constant([[y0]], dtype = 'float64')
        summation = []
        curr_I_nn = tf.constant([[sir_0[1]]], dtype = 'float64')
        curr_S_nn = tf.constant([[sir_0[0]]], dtype= 'float64')
        for i in range(N-1):
            next_y = curr_y + dt*g(curr_y, dataset[0, k, i])
            next_S_nn = curr_S_nn - dt*tf.matmul(curr_y, tf.matmul(curr_S_nn, curr_I_nn))
            next_I_nn = curr_I_nn + dt*(tf.matmul(curr_y, tf.matmul(curr_S_nn, curr_I_nn))) - dt*a*curr_I_nn
            if i % step_summation == 0:
                summation.append( ( ( next_I_nn - I[k, i+1] )/TOT )**2 )  
            curr_y = next_y
            curr_S_nn = next_S_nn
            curr_I_nn = next_I_nn
        total_summation.append(tf.reduce_sum(summation))
        #print(tf.reduce_sum(total_summation))
    return tf.reduce_mean(total_summation)
    

def train_step(K, dataset):
    with tf.GradientTape() as tape:
        loss = custom_loss(K, dataset)
    trainable_variables=list(weights.values())+list(biases.values())
    gradients = tape.gradient(loss, trainable_variables)
    optimizer.apply_gradients(zip(gradients, trainable_variables))
    
# override del numero massimo di iterazioni
    
if args.iterations == 0:
    training_steps = int(data_dict['training_steps'])
else:
    training_steps = args.iterations
display_step = int(data_dict['display_step'])

#####################
# training della rete
#####################

if args.train:
    print("Starting the training...\n")
    loss_history = np.zeros(int(training_steps/display_step))
    i_history = 0
    if args.validate:
        loss_history_val = np.zeros(int(training_steps/display_step))
    
    try:
        for i in range(training_steps):
          train_step(K, dataset)
          if i % display_step == 0:
            print("iterazione %i:" % i)
            loss_history[i_history] = custom_loss(K, dataset)
            print("loss on training set: %f " % loss_history[i_history])
            if args.validate:
                loss_history_val[i_history] = custom_loss(K_val, val_set)
                print("loss on validation set: %f" % loss_history_val[i_history])
            i_history = i_history + 1
    except KeyboardInterrupt:
        print('\nTraining interrupted by user. Proceeding to save the weights and plot the solutions...\n')

########################      
# i pesi vengono salvati
########################

if len(args.save_weights) > 0:
    print("Saving the weights in "+args.save_weights+"...\n")
    with open(args.save_weights, 'wb') as file:
        pickle.dump((weights, biases, dataset), file)

######################
# vengono fatti i plot
######################

# plot del test set
if not args.load_temp:
    n_plots = args.plot
else:
    n_plots = K_test

for p in range(n_plots):
    # check override fun-type
    if len(args.fun_type) == 0:
        T = tg.generate_temperature(data_dict["temperature_type"], t_max = t_max)
    else:
        T = tg.generate_temperature(args.fun_type, t_max = t_max)
    # inizializza i beta
    y_real = np.zeros(N)
    y_nn = np.zeros(N)
    y_real[0] = y0
    y_nn[0] = y0 
    curr_y = tf.constant([[y0]], dtype = 'float64')
    # inizializza le S, I
    I_real = np.zeros(N)
    I_nn = np.zeros(N)
    I_real[0] = I0
    I_nn[0] = I0 
    curr_I_nn = tf.constant([[I0]], dtype = 'float64')
    curr_S_nn = tf.constant([[S0]], dtype = 'float64')
    
    for i in range(N-1):
        if not args.load_temp:
            curr_temp = T(t[i])
        else: # se le temperature sono state caricate non viene usata la T generata casualmente
            curr_temp = test_set[0, p, i]
        next_y = curr_y + dt*g(curr_y, curr_temp)
        y_nn[i+1] = next_y.numpy()[0][0]
        y_real[i+1] = y_real[i] + dt*f(y_real[i], curr_temp)
        next_S_nn = curr_S_nn - dt*tf.matmul(curr_y, tf.matmul(curr_S_nn, curr_I_nn))
        next_I_nn = curr_I_nn + dt*(tf.matmul(curr_y, tf.matmul(curr_S_nn, curr_I_nn))) - dt*a*curr_I_nn
        I_nn[i+1] = next_I_nn.numpy()[0][0] 
        curr_y = next_y
        curr_S_nn = next_S_nn
        curr_I_nn = next_I_nn
    if p % 5 == 0:
        # viene calcolata la I_real
        s, I_real, r = hrk.RungeKutta(sir_0, y_real, N, t_max, a)
        # plot dei beta
        plt.plot(t, y_real)
        plt.plot(t, y_nn)
        plt.legend(["soluzione reale", "soluzione rete"])
        plt.title('beta, con test set {}'.format(p+1))
        plt.show()
        # plot delle I
        plt.plot(t, I_real)
        plt.plot(t, I_nn)
        plt.legend(["soluzione reale", "soluzione rete"])
        plt.title('infetti, con test set {}'.format(p+1))
        plt.show()
    
    

# plot del training set

if args.plot_train:
    for k in range(K):
        y_nn = np.zeros(N)
        y_nn[0] = y0 
        curr_y = tf.constant([[y0]], dtype = 'float64')
        # inizializza le S, I
        I_real = np.zeros(N)
        I_nn = np.zeros(N)
        I_real[0] = I0
        I_nn[0] = I0 
        curr_I_nn = tf.constant([[I0]], dtype = 'float64')
        curr_S_nn = tf.constant([[S0]], dtype = 'float64')
        # vengono calcolate le I e i beta
        for i in range(N-1):
            next_y = curr_y + dt*g(curr_y, dataset[0, k, i])
            y_nn[i+1] = next_y.numpy()[0][0]
            next_S_nn = curr_S_nn - dt*tf.matmul(curr_y, tf.matmul(curr_S_nn, curr_I_nn))
            next_I_nn = curr_I_nn + dt*(tf.matmul(curr_y, tf.matmul(curr_S_nn, curr_I_nn))) - dt*a*curr_I_nn
            I_nn[i+1] = next_I_nn.numpy()[0][0] 
            curr_y = next_y
            curr_S_nn = next_S_nn
            curr_I_nn = next_I_nn
        if k % 5 == 0:
            # viene calcolata la I_real
            s, I_real, r = hrk.RungeKutta(sir_0, dataset[1, k, :], N, t_max, a)
            # plot dei beta
            plt.plot(t, dataset[1, k, :])
            plt.plot(t, y_nn)
            plt.legend(["soluzione reale", "soluzione rete"])
            plt.title('beta, con training set {}'.format(k+1))
            plt.show()
            # plot delle I
            plt.plot(t, I_real)
            plt.plot(t, I_nn)
            plt.legend(["soluzione reale", "soluzione rete"])
            plt.title('infetti, con training set {}'.format(k+1))
            plt.show()

# plot della loss

it = np.arange(0, training_steps, display_step)
plt.plot(it, loss_history)
plt.plot(it, loss_history_val)
plt.legend(["loss training set", "loss validation set"])
plt.title('evoluzione della loss')
plt.show()






