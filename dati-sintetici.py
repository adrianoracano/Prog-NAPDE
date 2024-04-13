import pickle
from srcs import extract_infects as exi
from srcs import extract_removed as exr
import math
import re
import numpy as np
from matplotlib import pyplot as plt
from scipy.interpolate import make_interp_spline
from utilities import SirHelperFunctions as shr
import os
import converti_ngiorni_data as cgd
import matplotlib.animation as animation

###caricamento del dataset real===================================
path_i = "COVID-19/dati-regioni"
n_mesi = 18
n_giorni = math.floor(n_mesi * 30)
n_timesteps = 30 * n_mesi #ora n_timestep può essere scelto grande a piacere

date = ['15/03/2020','29/08/2020', '15/09/2020', '30/09/2020', '15/10/2020', '30/10/2020', '15/11/2020', '30/11/2020']  # le date devono essere stringhe nella forma 'dd/mm/yyyy', successive al 24 feb 2020 e precedenti il 31 dic 2020
data = date[0]

#caricamento infetti =============================================
infetti, beta_log = exi.extract_infects(path_i, n_timesteps, data, n_mesi)
rimossi =  exr.extract_removed(path_i, n_timesteps, data, n_mesi)

#smoothing infetti ===============================================
smooth = False
if smooth :
    # Inizializzazione dell'array di output
    I_vec = infetti
    K = 1 / 2 #percentuale dei timestep usati per interpolare gli infetti, si moltiplica a quello in exi
    n_interp_point = max(10,math.floor(n_giorni * K)) #numero punti in cui interpolare gli infetti
    new_indices = np.linspace(0, I_vec.shape[1] - 1, n_interp_point)
    inf_interp = np.zeros(shape=(I_vec.shape[0], n_interp_point))
    spline_indices = np.linspace(0, I_vec.shape[1] - 1, n_timesteps + 1)
    inf_spline = np.zeros(shape=(I_vec.shape[0], n_timesteps + 1))

#beta sintetico ==================================================
use_positive = True #per tagliare beta_log negativo
if use_positive:
    beta_log = beta_log.clip(min = 0)
up_bound = 20 #per tagliare beta sopra una soglia
use_up_bound = True
if use_up_bound:
    beta_log = beta_log.clip(max = up_bound)
K = 1 / 20 # percentule di timestep per interpolare beta, regola lo smoothing di beta
n_interp_points = math.floor((n_timesteps) * K)
new_indices = np.linspace(0, beta_log.shape[1], n_interp_points)
beta_log_interp = np.zeros(shape=(beta_log.shape[0], n_interp_points))
spline_indices = np.linspace(0, beta_log.shape[1] - 1, n_timesteps)
beta_spline = np.zeros(shape=(beta_log.shape[0], n_timesteps))
# Interpolazione lineare (beta) lungo l'asse N per ogni riga 
for i in range(beta_log.shape[0]):
    beta_log_interp[i, :] = np.interp(new_indices, np.arange(beta_log.shape[1]), beta_log[i, :])
    spline = make_interp_spline(new_indices, beta_log_interp[i, :])
    beta_spline[i, :] = spline(spline_indices)

#======================= AGGIUNTA RUMORE =========================   
#tensore delle medie
mean = np.mean(beta_spline, axis = 1)
mean = mean.reshape(beta_spline.shape[0], 1)
np.tile(mean, beta_spline.shape[1])
#tensore delle ampiezze
amp = np.max(np.absolute(beta_spline - mean), axis = 1)
amp = amp.reshape(beta_spline.shape[0], 1)
np.tile(amp, beta_spline.shape[1])
#vettore tempo
t = np.arange(beta_spline.shape[1])/beta_spline.shape[1] * n_giorni
#aggiunta di rumore
noise = 0
sin = np.vectorize(math.sin)
#rumore 1(frequenza alta, ampiezza bassa)
omega = 2 * math.pi / (n_giorni * 0.05)
vec = sin(omega * t)
vec = vec.reshape(1, beta_spline.shape[1])
np.tile(vec, beta_spline.shape[1])
noise = noise + 0.05* amp * vec
#rumore 2(frequenza bassa, ampiezza alta)
vec2 = sin(omega/5 * t)
vec2 = vec2.reshape(1, beta_spline.shape[1])
np.tile(vec2, beta_spline.shape[1])
noise = noise + 0.15 * amp * vec2
beta_spline = beta_spline + noise

#generazione degli infetti tramite il SIR=========================
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

save_plots = ''

#vettore delle date sulle ascisse ================================
t = np.arange(infetti.shape[1])/infetti.shape[1] * n_giorni
t_date = []
t_spaced = []
t_date_spaced = []
giorni_prec = cgd.data_a_ngiorni(data)
for i in range(len(t)):
    t_date.append(cgd.ngiorni_a_data(t[i] + giorni_prec))
    if i % 100 == 0:
        t_spaced.append(t[i])
        t_date_spaced.append(cgd.ngiorni_a_data(math.floor(t[i]) + giorni_prec))

""" print(t_spaced);
print(t_date_spaced) """

#linee di accettazione di beta ===================================
zero = np.zeros(shape = t.shape)
upper_bound = zero + 20

#plot ============================================================
for i in range(19):
    fig, ax = plt.subplots(2,1, figsize = (16,8))
    ax[0].plot(t, beta_log[i,:])
    ax[0].plot(t, beta_spline[i,:])
    ax[1].plot(t, infetti[i,:])
    ax[1].plot(t, infetti_SIR[i,:])
    ax[0].legend(['beta log','beta sintetico'])
    ax[1].legend(['infetti reali', 'infetti SIR'])
    #ax[1].legend(['infetti SIR'])
    ax[0].plot(t, zero,'--')
    ax[0].plot(t, upper_bound,'--')
    ax[0].set_xticks(t_spaced)
    ax[0].set_xticklabels(t_date_spaced)
    ax[1].set_xticks(t_spaced)
    ax[1].set_xticklabels(t_date_spaced)
    plt.suptitle(str(reg_list[i]) + ', dal ' + data)
    plt.show()
    if len(save_plots) > 0:
        print("Saving the plots in " + save_plots + "...\n")
        path = "./" + save_plots
        if not os.path.exists(path):
            os.mkdir(path)
        filepath2 = path + "/" + str(reg_list[i]) + ".png";
        plt.savefig(fname=filepath2)


