from srcs import generate_beta_arrays as gba
from srcs import generate_temp_arrays as gta
import numpy as np
import pickle


# DATI DA SCEGLIERE
"""
. n_timesteps = 
. nome_file = 
.
"""

n_timesteps = 100
path_i = "COVID-19/dati-regioni"
K = 5
n_giorni = 15
overlap = 3
path_t = "Temperature"
nome_file = "prova_dataset_betalog_giorni.pkl"
K_train = 32 # il resto è messo nel validation set

# vengono estratti i dati
beta, infetti = gba.generate_beta_arrays(path_i, K, n_giorni, overlap, regions = ["Abruzzo"])
temp = gta.generate_temp_arrays(path_t, K, n_giorni, overlap, regions = ["Chieti"])

print(beta.shape)
print(temp.shape)

temp_train = temp[0:K_train, :]
beta_train = beta[0:K_train, :]
infetti_train = infetti[0:K_train, :]

K_val = temp.shape[0] - K_train
temp_val = temp[K_train:, :]
beta_val = beta[K_train:, :]
infetti_val = infetti[K_train:, :]
print(infetti_val.shape)

with open('datasets/'+nome_file, 'wb') as file:
    pickle.dump((infetti_train, infetti_val, temp_train, temp_val, beta_train, beta_val, n_giorni), file)
