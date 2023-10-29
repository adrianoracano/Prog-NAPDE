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
tf.keras.backend.set_floatx('float64')

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
parser.add_argument('-pt', '--plot-train', help='plot using the train set', action='store_true')
parser.add_argument('-lt', '--load-temp', help='load the temperatures', action='store_true')
parser.add_argument('-d', '--default', help='activate flags: --train, --validate, --load-temp', action='store_true')
parser.add_argument('-o', '--overwrite', help='save and load the weights from the specified file', default='')
parser.add_argument('-sp', '--save-plots', help='save the plots in the specified file after the training', default='')
parser.add_argument('-sm', '--save-model', help='save the model file in the specified directory after the training', default='')

args = parser.parse_args()

#######################
# stringa da inserire:
# 1) python prova-sir-keras.py --load-temp --new-model --train --validate --save-model saved-models/nomemodello --plot-train --save-plots saved-plots/nomemodello
# o
# 2) python prova-sir-keras.py --load-temp --load-model saved-models/nomemodello --train --validate --save-model saved-models/nomemodello --plot-train --save-plots saved-plots/nomemodello
#######################

if args.default:
    args.train = True
    args.validate = True
    args.load_temp = True

if len(args.overwrite) > 0:
    args.save_weights = args.overwrite
    args.load_weights = args.overwrite

if len(args.load_weights) > 0 and args.new_weights:
    print("Cannot generate new weights and loading existing ones. Aborting..\n")
    sys.exit()

# print-help da completare

if args.print_help:
    print('scrivere help per lo script')
    sys.exit()

###############################
# i dati vengono presi dal file
###############################

data_dict = {}

with open(args.file, 'r') as file:
    for line in file:
        line = line.strip()  # Remove leading/trailing whitespace
        if line:
            field, value = line.split(':')
            data_dict[field.strip()] = value.strip()

learning_rate = float(data_dict['learning_rate'])
n_input = 2
n_hidden = int(data_dict['n_hidden'])  # il numero di hidden neurons viene preso dal file data.txt
n_output = 1
display_step = int(data_dict['display_step'])
solver = data_dict['solver']  # il tipo di metodo usato per il training: ea oppure rk

####################################
# vengono generati o caricati i pesi
####################################
K_test = 0  # solo una variabile utile per i plot finali
# if len(args.load_weights) > 0:
#     print("Loading weights from " + args.load_weights + "...\n")
#     with open(args.load_weights, 'rb') as file:
#         weights, biases, dataset = pickle.load(file)
# N = int(data_dict['N'])  # in base a quanto vale N vengono caricate le temperature giuste

# if args.load_temp:
#     try:
#         nome_file_temp = 'LOAD_TEMP_N_'+data_dict['N']+'_K_'+data_dict['K']
#         if data_dict['mixed'] == 'yes':
#             nome_file_temp = nome_file_temp+'_MIXED.pkl'
#         else:
#             nome_file_temp = nome_file_temp+'.pkl'
#         nome_file_temp = 'datasets/'+nome_file_temp
#         with open(nome_file_temp, 'rb') as file:
#             dataset, K, val_set, K_val, test_set, K_test = pickle.load(file)  # viene caricato il  dataset
#         print('dataset', nome_file_temp, 'loaded...\n')
#     except FileNotFoundError:
#         print('file',nome_file_temp,'not found...\n')
#         sys.exit()

if args.load_temp:
    nome_file_temp = 'datasets/'+data_dict['dataset']
    try:
        with open(nome_file_temp, 'rb') as file:
            dataset, K, val_set, K_val, test_set, K_test = pickle.load(file)  # viene caricato il  dataset
        print('dataset', nome_file_temp, 'loaded...\n')
    except FileNotFoundError:
        print('file',nome_file_temp,'not found...\n')
        sys.exit()

# if args.new_weights:
#     print("Generating new weights...\n")
#     weights = {
#         'h1': tf.Variable(tf.random.normal([n_input, n_hidden], dtype='float64'), dtype='float64'),
#         'out': tf.Variable(tf.random.normal([n_hidden, n_output], dtype='float64'), dtype='float64')
#     }
#     biases = {
#         'b1': tf.Variable(tf.random.normal([n_hidden], dtype='float64'), dtype='float64'),
#         'out': tf.Variable(tf.random.normal([n_output], dtype='float64'), dtype='float64')
#     }

# Stochastic gradient descent optimizer.
# optimizer = tf.optimizers.SGD(learning_rate)
optimizer = tf.optimizers.Adam(learning_rate)

##############
# Create or load the model
##############
tfk = tf.keras
tfkl = tf.keras.layers
tf.keras.backend.set_floatx('float64')

seed = 100

random.seed(seed)
os.environ['PYTHONHASHSEED'] = str(seed)
np.random.seed(seed)
tf.random.set_seed(seed)
tf.compat.v1.set_random_seed(seed)

initializers = tf.keras.initializers
init = initializers.GlorotNormal(seed = seed)
tfkl.Normalization(dtype='float64')

if len(args.load_model) > 0 and args.new_model:
    print("Cannot generate new model and loading an existing one. Aborting..\n")
    sys.exit()

if len(args.load_model) > 0:
    print("Loading model from " + args.load_model + "...\n")
    model = tfk.models.load_model(args.load_model)
elif args.new_model:
    model = tfk.Sequential()
    model.add(tfkl.Dense(
        units=n_hidden,
        input_dim=n_input,
        activation=None,
        use_bias=True,
        kernel_initializer="glorot_uniform",
        bias_initializer="zeros",
        kernel_regularizer=None,
        bias_regularizer=None,
        activity_regularizer=None,
        kernel_constraint=None,
        bias_constraint=None))
    model.add(tfkl.BatchNormalization())
    model.add(tfkl.Activation('sigmoid'))
    #model.add(tfkl.Dropout(0.3))
    model.add(tfkl.Dense(1))
    model.add(tfkl.BatchNormalization())


def g(y, v):
    v.shape = (v.shape[0], 1)
    tv = tf.constant(v, dtype='float64')
    x = tf.concat([y, tv], 1)
    return model(x)

###################
# definizione b_ref
###################
alpha = float(data_dict['alpha'])
S0 = float(data_dict['S0'])
S_inf = float(data_dict['S_inf'])
b_ref = alpha*math.log(S0/S_inf)/(1-S_inf)

t_max = float(data_dict['t_max'])
y0 = float(data_dict['beta0'])
N = int(data_dict['N'])
t = np.linspace(0, 1.0, N)
dt = 1.0 / N
if not args.load_temp:  # se le temperature non sono state caricate, il num di temperature è letto da data.txt
    K = int(data_dict['K'])  # numero di temperature
    K_val = int(data_dict['K_val'])  # numero di temperature da usare nel validation set
temperature = []
temperature_val = []


# vengono generate le temperature per il training
tau = 0.2
def f(beta, Betaeq): # è la funzione che regola beta:   beta(t)' = f(beta(t), T(t))
    return (1/tau)*(Betaeq - beta)*t_max#betaeq va cambiato con la t_return di mytemperaturegenerator, fatto


data = {  # questo dict viene usato per generare il dataset
    'beta0': np.array([y0]),
    'f': f,
    't_max': t_max,
    'N': N
}

if len(args.load_weights) == 0 and not args.load_temp:  # se i pesi e le temp non sono stati caricati, vengono generate nuove temp
    print("Generating new temperatures for train set...\n")
    for k in range(K):
        if len(args.fun_type) == 0:  # override dei tipi di temperature
            T = tg.generate_temperature(data_dict["temperature_type"], t_max=t_max)
        else:
            T = tg.generate_temperature(args.fun_type, t_max=t_max)
        temperature.append(T)
    dataset = dsg.generate_dataset(temperature, data)

if args.validate and not args.load_temp:  # vengono generate le temperature da usare nel validation set
    print("Generating new temperatures for validation set...\n")
    for k in range(K_val):
        if len(args.fun_type) == 0:
            T_val = tg.generate_temperature(data_dict["temperature_type"], t_max=t_max)
        else:
            T_val = tg.generate_temperature(args.fun_type, t_max=t_max)
        temperature_val.append(T_val)
    val_set = dsg.generate_dataset(temperature_val, data)

# vengono generati le osservazioni degli infetti

I = np.zeros([K, N])  # da usare per gli infetti
if args.validate:
    I_val = np.zeros([K_val, N])  # da usare per il validation set
a = float(data_dict['alpha'])
S0 = float(data_dict['S0'])
I0 = float(data_dict['I0'])
R0 = float(data_dict['R0'])
TOT = S0 + I0 + R0  # popolazione totale
sir_0 = np.array([S0, I0, R0])
for k in range(K):
    s, i, r = hrk.RungeKutta(sir_0, dataset[1, k,], N, t_max, a)
    I[k,] = i.copy()
if args.validate:
    for k in range(K_val):
        s, i, r = hrk.RungeKutta(sir_0, val_set[1, k,], N, t_max, a)
        I_val[k,] = i.copy()

dt = 1.0 / N
step_summation = int(data_dict['step_summation'])



########################
# definizione della loss
########################

def custom_loss(K, dataset, I):
    summation = []
    curr_beta = tf.constant(y0 * np.ones([K, 1], dtype='float64'), dtype='float64')
    curr_I_nn = tf.constant(sir_0[1] * np.ones([K, 1], dtype='float64'), dtype='float64')
    curr_S_nn = tf.constant(sir_0[0] * np.ones([K, 1], dtype='float64'), dtype='float64')
    for i in range(N - 1):
        next_beta = curr_beta + dt * t_max * g(curr_beta/b_ref, dataset[0, :, i]/b_ref)
        if solver == 'ea':
            next_S_nn = curr_S_nn - t_max * dt * curr_beta * curr_S_nn * curr_I_nn
            next_I_nn = curr_I_nn + t_max * dt * curr_beta * curr_S_nn * curr_I_nn - dt * a * curr_I_nn
        if solver == 'rk':
            next_S_nn, next_I_nn = shf.runge_kutta_step(curr_S_nn, \
                                                        curr_I_nn, curr_beta, next_beta, dt, a)
        if i % step_summation == 0:
            I_exact = I[:, i + 1]
            I_exact.shape = (I_exact.shape[0], 1)
            summation.append(tf.reduce_mean((next_I_nn - I_exact) ** 2))
        curr_beta = next_beta
        curr_S_nn = next_S_nn
        curr_I_nn = next_I_nn
    return tf.reduce_sum(summation)


def train_step(K, dataset, I):
    with tf.GradientTape() as tape:
        loss = custom_loss(K, dataset, I)
    trainable_variables = model.trainable_variables
    gradients = tape.gradient(loss, trainable_variables)
    optimizer.apply_gradients(zip(gradients, trainable_variables))


# override del numero massimo di iterazioni

if args.iterations == 0:
    training_steps = int(data_dict['training_steps'])
else:
    training_steps = args.iterations
display_step = int(data_dict['display_step'])
display_weights = int(data_dict['display_weights'])

#####################
# training della rete
#####################
n_iter = 0
if args.train:
    print("Starting the training...\n")
    loss_history = np.zeros(int(training_steps / display_step))
    i_history = 0
    if args.validate:
        loss_history_val = np.zeros(int(training_steps / display_step))

    try:
        for i in range(training_steps):
            train_step(K, dataset, I)
            if i % display_step == 0:
                print("iterazione %i:" % i)
                loss_history[i_history] = custom_loss(K, dataset, I)
                print("loss on training set: %f " % loss_history[i_history])
                if args.validate:
                    loss_history_val[i_history] = custom_loss(K_val, val_set, I_val)
                    print("loss on validation set: %f" % loss_history_val[i_history])
                i_history = i_history + 1
            if i % display_weights == 0:
                print("pesi all'iterazione %i:")
                print(model.trainable_variables)
            n_iter = i
    except KeyboardInterrupt:
        print('\nTraining interrupted by user. Proceeding to save the weights and plot the solutions...\n')

########################
# i pesi vengono salvati
########################

# if len(args.save_weights) > 0:
#     print("Saving the weights in " + args.save_weights + "...\n")
#     model.save(args.save_weights)
#     with open(args.save_weights, 'wb') as file:
#         pickle.dump((weights, biases, dataset), file)

########################
# il modello viene salvato
########################

if len(args.save_model) > 0:
    print("Saving the model in " + args.save_model + "...\n")
    model.save(args.save_model + "iter" + str(n_iter))


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
        T = tg.generate_temperature(data_dict["temperature_type"], t_max=t_max)
    else:
        T = tg.generate_temperature(args.fun_type, t_max=t_max)
    # inizializza i beta
    y_real = np.zeros(N)
    y_nn = np.zeros(N)
    y_real[0] = y0
    y_nn[0] = y0
    curr_y = tf.constant([[y0]], dtype='float64')
    # inizializza le S, I
    I_real = np.zeros(N)
    I_nn = np.zeros(N)
    I_real[0] = I0
    I_nn[0] = I0
    curr_I_nn = tf.constant([[I0]], dtype='float64')
    curr_S_nn = tf.constant([[S0]], dtype='float64')

    for i in range(N - 1):
        if not args.load_temp:
            curr_temp = np.array([T(t[i])], dtype='float64')
        else:  # se le temperature sono state caricate non viene usata la T generata casualmente
            curr_temp = np.array([test_set[0, p, i]], dtype='float64')
            curr_betaeq = np.array([test_set[2, p, i]], dtype='float64')
        next_y = curr_y + t_max * dt*g(curr_y/b_ref, curr_temp/b_ref)
        y_nn[i + 1] = next_y.numpy()[0][0]
        y_real[i + 1] = y_real[i] + t_max * dt * f(y_real[i], curr_betaeq)
        # viene usato runge kutta oppure eulero in avanti per calcolare uno step della soluzione
        if solver == 'rk':
            next_S_nn, next_I_nn = shf.runge_kutta_step(curr_S_nn, curr_I_nn, \
                                                        curr_y, next_y, dt, a)
        if solver == 'ea':
            next_S_nn = curr_S_nn - t_max * dt * curr_y * curr_S_nn * curr_I_nn
            next_I_nn = curr_I_nn + t_max * dt * curr_y * curr_S_nn * curr_I_nn - dt * a * curr_I_nn
        I_nn[i + 1] = next_I_nn.numpy()[0][0]
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
        plt.title('beta, con test set {}'.format(p + 1))
        if len(args.save_plots) > 0:
            print("Saving the plots in " + args.save_plots + "...\n")
            path = "./" + args.save_plots + "iter" + str(n_iter);
            if not os.path.exists(path):
                os.mkdir(path)
            filepath11 = path + "/betatest" + str(p + 1) + ".png";
            plt.savefig(fname=filepath11)
        plt.close()
        plt.show()
        # plot delle I
        plt.plot(t, I_real)
        plt.plot(t, I_nn)
        plt.legend(["soluzione reale", "soluzione rete"])
        plt.title('infetti, con test set {}'.format(p + 1))
        if len(args.save_plots) > 0:
            print("Saving the plots in " + args.save_plots + "...\n")
            path = "./" + args.save_plots + "iter" + str(n_iter);
            if not os.path.exists(path):
                os.mkdir(path)
            filepath22 = path + "/infettitest" + str(p + 1) + ".png";
            plt.savefig(fname=filepath22)
        plt.close()
        plt.show()

# plot del training set

if args.plot_train:
    for k in range(K):
        y_nn = np.zeros(N)
        y_nn[0] = y0
        curr_y = tf.constant([[y0]], dtype='float64')
        # inizializza le S, I
        I_real = np.zeros(N)
        I_nn = np.zeros(N)
        I_real[0] = I0
        I_nn[0] = I0
        curr_I_nn = tf.constant([[I0]], dtype='float64')
        curr_S_nn = tf.constant([[S0]], dtype='float64')
        # vengono calcolate le I e i beta
        for i in range(N - 1):
            T_curr = np.array([dataset[0, k, i]], dtype='float64')
            next_y = curr_y + t_max * dt * g(curr_y/b_ref, T_curr/b_ref)
            y_nn[i + 1] = next_y.numpy()[0][0]
            if solver == 'rk':
                next_S_nn, next_I_nn = shf.runge_kutta_step(curr_S_nn, curr_I_nn, \
                                                            curr_y, next_y, dt, a)
            if solver == 'ea':
                next_S_nn = curr_S_nn - t_max * dt * curr_y * curr_S_nn * curr_I_nn
                next_I_nn = curr_I_nn + t_max * dt * curr_y * curr_S_nn * curr_I_nn - dt * a * curr_I_nn
            curr_y = next_y
            curr_S_nn = next_S_nn
            curr_I_nn = next_I_nn
            I_nn[i + 1] = next_I_nn.numpy()[0][0]
        if k % 5 == 0:
            # viene calcolata la I_real
            s, I_real, r = hrk.RungeKutta(sir_0, dataset[1, k, :], N, t_max, a)
            # plot dei beta
            plt.plot(t, dataset[1, k, :])
            plt.plot(t, y_nn)
            plt.legend(["soluzione reale", "soluzione rete"])
            plt.title('beta, con training set {}'.format(k + 1))
            if len(args.save_plots) > 0:
                print("Saving the plots in " + args.save_plots + "...\n")
                path = "./" + args.save_plots + "iter" + str(n_iter);
                if not os.path.exists(path):
                    os.mkdir(path)
                filepath1 = path + "/betatrain" + str(k + 1) + ".png";
                plt.savefig(fname=filepath1)
            plt.close()
            # if args.save_plots:
            #     filepath1 = "saved-plots/beta" + str(k) + ".png";
            #     plt.savefig(fname=filepath1)
            # plt.show()
            # plot delle I
            plt.plot(t, I_real)
            plt.plot(t, I_nn)
            plt.legend(["soluzione reale", "soluzione rete"])
            plt.title('infetti, con training set {}'.format(k + 1))
            if len(args.save_plots) > 0:
                print("Saving the plots in " + args.save_plots + "...\n")
                path = "./" + args.save_plots + "iter" + str(n_iter);
                if not os.path.exists(path):
                    os.mkdir(path)
                filepath2 = path + "/infettitrain" + str(k + 1) + ".png";
                plt.savefig(fname=filepath2)
            plt.close()
            # if args.save_plots:
            #     filepath2 = "saved-plots/infetti" + str(k) + ".png";
            #     plt.savefig(fname=filepath2)
            # plt.show()

# plot della loss

it = np.arange(0, training_steps, display_step)
plt.plot(it, loss_history)
plt.plot(it, loss_history_val)
plt.legend(["loss training set", "loss validation set"])
plt.title('evoluzione della loss')
if len(args.save_plots) > 0:
    print("Saving the loss plot in " + args.save_plots + "...\n")
    path = "./" + args.save_plots + "iter" + str(n_iter);
    if not os.path.exists(path):
        os.mkdir(path)
    filepath3 = path + "/lossevolution" + ".png";
    plt.savefig(fname=filepath3)
plt.close()
plt.show()





