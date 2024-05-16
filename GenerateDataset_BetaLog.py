from srcs import generate_beta_arrays as gba
from srcs import generate_temp_arrays as gta
from srcs import generate_zone_arrays as gza
import numpy as np
import pickle
import math

# DATI DA SCEGLIERE

n_timesteps = 100
path_i = "COVID-19/dati-regioni"
regione_beta = ['Emilia-Romagna']
K = 40
n_giorni = 15
overlap = 3
path_t = "Temperature"
regione_temp = ['Bologna']
K_train = math.floor(K*0.8) # il resto è messo nel validation set

use_zone = True

# vengono estratti i dati
beta, infetti, date = gba.generate_beta_arrays_2(path_i, K, n_giorni, overlap, regions = regione_beta)
temp = gta.generate_temp_arrays_2(path_t, K, n_giorni, overlap, regions = regione_temp)

# print(beta.shape)
# print(temp.shape)

val_at_end = False

if val_at_end:
    ### vecchia parte che mette validation alla fine

    temp_train = temp[0:K_train, :]
    beta_train = beta[0:K_train, :]
    infetti_train = infetti[0:K_train, :]
    date_train = date[0:K_train]

    K_val = K - K_train
    temp_val = temp[K_train:, :]
    beta_val = beta[K_train:, :]
    infetti_val = infetti[K_train:, :]
    date_val = date[K_train:]
    print(infetti_val.shape)

else:

    ### ora vogliamo validation un po' sparsi
    K_val = K - K_train
    date_val = ['1/06/2020', '10/07/2020', '13/12/2020', '3/02/2021', '9/04/2021', '13/06/2021', '4/08/2021', '17/08/2021']
    val_indexes = np.where(np.isin(date, date_val))[0]
    temp_val = temp[val_indexes]
    beta_val = beta[val_indexes]
    infetti_val = infetti[val_indexes]

    #date_train = date[~np.isin(np.arange(len(date)), np.array(val_indexes, dtype=object))]
    date_train = [elem for elem in date if elem not in date_val]
    train_indexes = np.where(np.isin(date, date_train))[0]
    temp_train = temp[train_indexes]
    beta_train = beta[train_indexes]
    infetti_train = infetti[train_indexes]

#nome_file = "Prova_Bologna.pkl"
nome_file = regione_beta[0] + '_' + str(K_train) + 'train_' + str(K_val) + 'val'
if not val_at_end:
    nome_file += '_sparse'

nome_file += '.pkl'

if not use_zone:
    with open('datasets/'+nome_file, 'wb') as file:
        pickle.dump((infetti_train, infetti_val, temp_train, temp_val, beta_train, beta_val, n_giorni, date_train, date_val), file)
else:

    zone = gza.generate_zone_arrays_2(K, n_giorni, overlap, regions=regione_beta)
    if val_at_end:
        zone_train = np.expand_dims(zone[0:K_train, :], axis = -1)
        zone_val = np.expand_dims(zone[K_train:, :], axis = -1)
    else:
        zone_train = np.expand_dims(zone[train_indexes], axis=-1)
        zone_val = np.expand_dims(zone[val_indexes], axis=-1)

    temp_train = np.expand_dims(temp_train, axis = -1)
    temp_val = np.expand_dims(temp_val, axis = -1)
    train_set = np.concatenate((temp_train, zone_train), axis = -1)
    val_set = np.concatenate((temp_val, zone_val), axis = -1)

    with open('datasets/'+nome_file, 'wb') as file:
        pickle.dump((infetti_train, infetti_val, train_set, val_set, beta_train, beta_val, n_giorni, date_train, date_val), file)
