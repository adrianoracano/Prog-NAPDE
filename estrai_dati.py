import math

from srcs import extract_infects as exi
from srcs import extract_temperatures as ext
from srcs import extract_zone as exz
import numpy as np
import pickle

###################
# dati da scegliere
###################

# scrivere il percorso delle cartelle
path_t = "Temperature"
path_i = "COVID-19/dati-regioni"

# scegliere il nome del file
nome_file = "NUOVO_DATASET.pkl"

# valore sommato agli infetti iniziali
eps_I0 = 1e-4

# scegliere il numero di timesteps (se lo si vuole diminuire)
n_timesteps = math.floor(365 / 3) # non può essere maggiore di questo

# scegliere il numero di regioni per il training set (il resto è validation set)
K_train = 10


# scegliere se costruire il dataset con temperature e zone
temps_and_lockdown = True

# segliere i valori di beta0
# beta0 = np.repeat(2.5, 19)
# Rt = [3.14 , 2.99, 3.31, 2.32, 1.96, 2.39 ]
# per rendere le temperature regolari. 0 vuol dire che le temperature
# non vengono cambiate
smooth_param = 0

#######################
# fine dati da scegliere
#######################

start_vec_train = ['24/03/2020' , '24/05/2020', '24/07/2020', '24/09/2020', '24/11/2020']  # le date devono essere stringhe nella forma 'dd/mm/yyyy', successive al 24 feb 2020 e precedenti il 31 dic 2020
n_date_train = len(start_vec_train)
n_mesi = 4;

start = start_vec_train[0]
temperature_train = ext.extract_temperatures(path_t,  n_timesteps, start, n_mesi)
infetti_train, beta0_train = exi.extract_infects(path_i, n_timesteps, start, n_mesi)
zone_train = exz.extract_zones(n_timesteps, start, n_mesi)

for start in start_vec_train[1:]:
    temperature_train = np.concatenate((temperature_train, ext.extract_temperatures(path_t,  n_timesteps, start, n_mesi)), axis = 0)
    infetti_new, beta_new = exi.extract_infects(path_i, n_timesteps, start, n_mesi)
    infetti_train = np.concatenate((infetti_train, infetti_new), axis = 0)
    beta0_train = np.concatenate((beta0_train, beta_new), axis = 0)
    zone_train = np.concatenate((zone_train, exz.extract_zones(n_timesteps, start, n_mesi)), axis = 0)

infetti_train[:, 0] = infetti_train[:, 0] + eps_I0
S0_train = 1 - infetti_train[:,0]
# beta0 = Rt * alpha / S0;

start_vec_val = ['10/02/2021' , '10/04/2021', '10/06/2021']  # le date devono essere stringhe nella forma 'dd/mm/yyyy', successive al 24 feb 2020 e precedenti il 31 dic 2020
n_date_val = len(start_vec_val)

start = start_vec_val[0]
temperature_val = ext.extract_temperatures(path_t,  n_timesteps, start, n_mesi)
infetti_val, beta0_val = exi.extract_infects(path_i, n_timesteps, start, n_mesi)
zone_val = exz.extract_zones(n_timesteps, start, n_mesi)

for start in start_vec_val[1:]:
    temperature_val = np.concatenate((temperature_val, ext.extract_temperatures(path_t,  n_timesteps, start, n_mesi)), axis = 0)
    infetti_new, beta_new = exi.extract_infects(path_i, n_timesteps, start, n_mesi)
    infetti_val = np.concatenate((infetti_val, infetti_new), axis = 0)
    beta0_val = np.concatenate((beta0_val, beta_new), axis = 0)
    zone_val = np.concatenate((zone_val, exz.extract_zones(n_timesteps, start, n_mesi)), axis = 0)

infetti_val[:, 0] = infetti_val[:, 0] + eps_I0
S0_val = 1 - infetti_val[:,0]


# n_date = len(start_vec)
# K_train = K_train * n_date;

def replace_nan_with_previous(arr):
    # Trova le posizioni dei valori 'nan'
    nan_positions = np.isnan(arr)

    # Trova gli indici dei valori 'nan'
    nan_indices = np.argwhere(nan_positions)

    # Sostituisci i valori 'nan' con i valori precedenti lungo l'asse delle colonne
    for idx in nan_indices:
        row, col = idx
        if col != 0:  # Controlla se non è la prima colonna
            arr[row, col] = arr[row, col - 1]
    return arr

temperature_train = replace_nan_with_previous(temperature_train.copy())
temperature_val = replace_nan_with_previous(temperature_val.copy())


if smooth_param != 0:
    temps_smooth_train = np.zeros([19 * n_date_train, n_timesteps])
    temps_smooth_val = np.zeros([19 * n_date_val, n_timesteps])
    for i in range(n_timesteps):
        mean_train = temperature_train[:, i].copy()
        mean_val = temperature_val[:, i].copy()
        j = 1
        n_sum = 1
        while j < smooth_param:
            if i + j < n_timesteps:
                n_sum = n_sum + 1
                mean_train = mean_train + temperature_train[:, i+j]
                mean_val = mean_val + temperature_val[:, i+j]

            if i - j - 1 >= 0:
                mean_train = mean_train + temperature_train[:, i-j-1]
                mean_val = mean_val + temperature_val[:, i-j-1]
                n_sum = n_sum +1
            j =j+1
        temps_smooth_train[:, i] = (1.0/n_sum)* mean_train
        temps_smooth_val[:, i] = (1.0 / n_sum) * mean_val
    
    temperature_train = temps_smooth_train
    temperature_val = temps_smooth_val



# inf_train = infetti[0:K_train, :]
# inf_val = infetti[K_train:]
#
# beta0_train = beta0[0:K_train]
# beta0_val = beta0[K_train:]

if not temps_and_lockdown: 
    # train_set = temperature[0:K_train, :]
    # val_set = temperature[K_train:]
    train_set = temperature_train
    val_set = temperature_val
if temps_and_lockdown:
    # K_val = 19 * n_date - K_train
    train_set = np.zeros([n_date_train * 19, n_timesteps, 2])
    val_set = np.zeros([n_date_val * 19, n_timesteps, 2])
    train_set[:, :, 0] = temperature_train
    train_set[:, :, 1] = zone_train
    val_set[:, :, 0] = temperature_val
    val_set[:, :, 1] = zone_val


with open('datasets/'+nome_file, 'wb') as file:
    pickle.dump((infetti_train, infetti_val, train_set, val_set, beta0_train, beta0_val), file)
