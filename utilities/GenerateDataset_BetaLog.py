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
K_train = 15 # il resto è messo nel validation set

# vengono estratti i dati
beta, infetti = gba.generate_beta_arrays(path_i, K, n_giorni, overlap)
temp = gta.generate_temp_arrays(path_t, n_giorni, max_months, overlap)

temp_train = np.zeros((K_train, n_timesteps))
beta_train = np.zeros((K_train, n_timesteps))
infetti_train = np.zeros((K_train, n_timesteps))

K_val = temp.shape[0] - K_train
temp_val = np.zeros((K_val, n_timesteps))
beta_val= np.zeros((K_val, n_timesteps))
infetti_val = np.zeros((K_val, n_timesteps))

with open('datasets/'+nome_file, 'wb') as file:
    pickle.dump((infetti_train, infetti_val, temp_train, temp_val, beta_train, beta_val), file)
