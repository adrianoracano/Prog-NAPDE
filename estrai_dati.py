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
nome_file = "PROVA_VERA_3date.pkl"

# valore sommato agli infetti iniziali
eps_I0 = 1e-4

# scegliere il numero di timesteps (se lo si vuole diminuire)
n_timesteps = 312

# scegliere il numero di regioni per il training set (il resto è validation set)
K_train = 10


# scegliere se costruire il dataset con temperature e zone
temps_and_lockdown = True

# segliere i valori di beta0
beta0 = np.repeat(2.5, 19)
# Rt = [3.14 , 2.99, 3.31, 2.32, 1.96, 2.39 ]
# per rendere le temperature regolari. 0 vuol dire che le temperature
# non vengono cambiate
smooth_param = 0

#######################
# fine dati da scegliere
#######################

start_vec = ['18/03/2020' , '24/04/2020', '24/05/2020']  # le date devono essere stringhe nella forma 'dd/mm/yyyy', successive al 24 feb 2020 e precedenti il 31 dic 2020

start = start_vec[0]
temperature = ext.extract_temperatures(path_t,  n_timesteps, start)
infetti = exi.extract_infects(path_i, n_timesteps, start)
zone = exz.extract_zones(n_timesteps, start)

for start in start_vec[1:]:
    temperature = np.concatenate((temperature, ext.extract_temperatures(path_t,  n_timesteps, start)), axis = 0)
    infetti = np.concatenate((infetti, exi.extract_infects(path_i, n_timesteps, start)), axis = 0)
    zone = np.concatenate((zone, exz.extract_zones(n_timesteps, start)), axis = 0)

infetti[:, 0] = infetti[:, 0] + eps_I0
S0 = 1 - infetti[:,0]
# beta0 = Rt * alpha / S0;


n_date = len(start_vec)
K_train = K_train * n_date;

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

temperature = replace_nan_with_previous(temperature.copy())

if smooth_param != 0:
    temps_smooth = np.zeros([19, n_timesteps])
    for i in range(n_timesteps):
        mean = temperature[:, i].copy()
        j = 1
        n_sum = 1
        while j < smooth_param:
            if i + j < n_timesteps:
                n_sum = n_sum + 1
                mean = mean + temperature[:, i+j]
            if i - j - 1 >= 0:
                mean = mean + temperature[:, i-j-1]
                n_sum = n_sum +1
            j =j+1
        temps_smooth[:, i] = (1.0/n_sum)*mean
    
    temperature = temps_smooth


inf_train = infetti[0:K_train, :]
inf_val = infetti[K_train:]

beta0_train = beta0[0:K_train]
beta0_val = beta0[K_train:]

if not temps_and_lockdown: 
    train_set = temperature[0:K_train, :]
    val_set = temperature[K_train:]
if temps_and_lockdown:
    K_val = 19 * n_date - K_train
    train_set = np.zeros([K_train, n_timesteps, 2])
    val_set = np.zeros([K_val, n_timesteps, 2])
    train_set[:, :, 0] = temperature[0:K_train, :]
    train_set[:, :, 1] = zone[0:K_train, :]
    val_set[:, :, 0] = temperature[K_train:]
    val_set[:, :, 1] = zone[K_train:]


with open('datasets/'+nome_file, 'wb') as file:
    pickle.dump((inf_train, inf_val, train_set, val_set, beta0_train, beta0_val), file)
