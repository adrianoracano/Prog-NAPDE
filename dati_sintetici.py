import pickle
from srcs import extract_infects as exi
from srcs import extract_removed as exr
import math
import re
import numpy as np
from matplotlib import pyplot as plt
from scipy.interpolate import make_interp_spline
from utilities import SirHelperFunctions as shr
###caricamento del dataset real

path_i = "COVID-19/dati-regioni"
n_timesteps = 200 #ora n_timestep può essere scelto grande a piacere
n_mesi = 0.5


date = ['29/02/2020', '15/03/2020', '30/03/2020', '15/04/2020', '30/04/2020', '15/05/2020', '30/05/2020']  # le date devono essere stringhe nella forma 'dd/mm/yyyy', successive al 24 feb 2020 e precedenti il 31 dic 2020

infetti, beta_log = exi.extract_infects(path_i, n_timesteps, date[2], n_mesi)
rimossi =  exr.extract_removed(path_i, n_timesteps, date[2], n_mesi)

#beta sintetico
new_indices = np.linspace(0, beta_log.shape[1], math.floor((n_timesteps)/20))
beta_log_interp = np.zeros(shape=(beta_log.shape[0], math.floor((n_timesteps)/20)))
spline_indices = np.linspace(0, beta_log.shape[1] - 1, n_timesteps)
beta_spline = np.zeros(shape=(beta_log.shape[0], n_timesteps))

# Interpolazione lineare lungo l'asse N per ogni riga
for i in range(beta_log.shape[0]):
    beta_log_interp[i, :] = np.interp(new_indices, np.arange(beta_log.shape[1]), beta_log[i, :])
    #       beta_interp[i, :] = np.interp(new_indices, np.arange(I_vec.shape[1]), I_vec[i, :])
    spline = make_interp_spline(new_indices, beta_log_interp[i, :])
    beta_spline[i, :] = spline(spline_indices)

#aggiunta di rumore
n_giorni = math.floor(n_mesi * 30)
mean = np.mean(beta_spline, axis = 1)
mean = mean.reshape(beta_spline.shape[0], 1)
np.tile(mean, beta_spline.shape[1])
amp = np.max(np.absolute(beta_spline - mean), axis = 1)
amp = amp.reshape(beta_spline.shape[0], 1)
np.tile(amp, beta_spline.shape[1])
omega = 2 * math.pi / (n_giorni * 0.05)
t = np.arange(beta_spline.shape[1])/beta_spline.shape[1] * n_giorni
sin = np.vectorize(math.sin)
vec = sin(omega * t)
vec = vec.reshape(1, beta_spline.shape[1])
np.tile(vec, beta_spline.shape[1])
noise = 0.1 * amp * vec
beta_spline = beta_spline + noise

# #plot di prova beta logaritmico e beta sintetico
# fig, ax = plt.subplots(1,1, figsize = (6,6))
# ax.plot(t, beta_log[0,:])
# ax.plot(t, beta_spline[0,:])
# ax.legend(['beta log','beta sintetico'])
# plt.title('Abruzzo, dal ' + date[2])
# #plt.show()

#generazione degli infetti tramite il SIR
R0 = rimossi[:,0]
I0 = infetti[:,0]
S0 = 1 - I0 - R0
sir0 = (S0, I0, R0)
alpha = 1.3
infetti_SIR = shr.compute_I([beta_spline], n_mesi, alpha, sir0)
infetti_SIR = infetti_SIR[0]

reg_list = ['Abruzzo',
     'Basilicata',
     'Calabria',
     'Campania',
     'Emilia-Romagna',
     'Friuli Venezia Giulia',
     'Lazio',
     'Liguria',
     'Lombardia',
     'Marche',
     'Piemonte',
     'Puglia',
     'Sardegna',
     'Sicilia',
     'Toscana',
     'Umbria',
     'Veneto',
     'P.A. Bolzano',
     'P.A. Trento']

#confronto tra infetti reali e quelli generati con SIR(Abruzzo)
t = np.arange(infetti.shape[1])/infetti.shape[1] * n_giorni
for i in range(19):
    fig, ax = plt.subplots(2,1, figsize = (8,4))
    ax[0].plot(t, beta_log[i,:])
    ax[0].plot(t, beta_spline[i,:])
    #ax[1].plot(t, infetti[i,:])
    ax[1].plot(t, infetti_SIR[i,:])
    ax[0].legend(['beta log','beta sintetico'])
    ax[1].legend(['infetti SIR'])
    plt.suptitle(str(reg_list[i]) + ', dal ' + date[2])
    plt.show()

