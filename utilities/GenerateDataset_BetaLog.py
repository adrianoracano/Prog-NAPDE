import sys
sys.path.append('../srcs')
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

n_timesteps = 150
path_i = "../dati-regioni"
K = 25
n_giorni = 25
overlap = 5
path_t = "../prova-estrazione-temp/Temperature"
nome_file = "betalog.pkl"
K_train = 20 # il resto è messo nel validation set

# vengono estratti i dati
beta, infetti = gba.generate_beta_arrays(path_i, K, n_giorni, overlap, regions = ["Lazio"])
temp = gta.generate_temp_arrays(path_t, K, n_giorni, overlap, regions = ["Roma"])

print(beta.shape)
print(temp.shape)

temp_train = temp[0:K_train, :]
beta_train = beta[0:K_train, :]
infetti_train = infetti[0:K_train, :]

K_val = temp.shape[0] - K_train
temp_val = temp[K_train:, :]
beta_val = beta[K_train:, :]
infetti_val = infetti[K_train:, :]

with open('datasets/'+nome_file, 'wb') as file:
    pickle.dump((infetti_train, infetti_val, temp_train, temp_val, beta_train, beta_val), file)
