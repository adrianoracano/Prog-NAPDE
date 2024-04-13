import math
import pandas as pd
import numpy as np
import glob
import re
from scipy.interpolate import make_interp_spline

pd.options.display.max_columns = 21

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

def extract_infects(path, n_timesteps, start, n_mesi, regions = reg_list):
    dir_path = path
    # ngiorni = len([entry for entry in os.listdir(dir_path) if os.path.isfile(os.path.join(dir_path, entry))])
    ngiorni = 366 * 2

    nreg = 21
    all_files = sorted(glob.glob(dir_path + "/*.csv"))
    df = pd.read_csv(all_files[0])

    reg_list = list(df['denominazione_regione'])
    reg_list.remove('P.A. Trento')
    reg_list.remove('P.A. Bolzano')
    reg_list.insert(len(reg_list), 'P.A. Bolzano')
    reg_list.insert(len(reg_list), 'P.A. Trento')
    nomi_regioni = dict.fromkeys(reg_list, 0)
    df = df.reindex(columns=reg_list)
    tot_regione = {
        'Lombardia': 9950742,
        'Lazio': 5707112,
        'Campania': 5592175,
        'Veneto': 4838253,
        'Sicilia': 4802016,
        'Emilia-Romagna': 4426929,
        'Piemonte': 4240736,
        'Puglia': 3900852,
        'Toscana': 3651152,
        'Calabria': 1841300,
        'Sardegna': 1575028,
        'Liguria': 1502624,
        'Marche': 1480839,
        'Abruzzo': 1269860,
        'Friuli Venezia Giulia': 1192191,
        'Umbria': 854137,
        'Basilicata': 536659,
        'Molise': 289840,
        "Valle d'Aosta": 122955,
        'P.A. Trento': 542158,
        'P.A. Bolzano': 532616}
    tot_regione = pd.Series(tot_regione)

    # dfi.loc[0] = pd.Series(np.array(df['totale_positivi']), index=dfi.columns)

    # giorno = 1
    dfi = pd.DataFrame(data=nomi_regioni, index=[0])
    giorno = 0
    start_vec = [0];  # giorno da cui iniziare a registrare infetti
    for file in (all_files[start_vec[0]: start_vec[0] + ngiorni]):
        df = pd.read_csv(file)
        dfi.loc[giorno] = pd.Series(np.array(df['totale_positivi']), index=dfi.columns)
        # infetti[giorno][:] = (np.array(df['totale_positivi']))
        giorno = giorno + 1

    dfi = dfi / tot_regione
    dfi = dfi.reindex(columns=reg_list)
    #dfi = dfi.drop(columns=['Molise', "Valle d'Aosta"])

    for region in reg_list:
        if not region in regions:
            dfi = dfi.drop(columns=[region])

    n_giorni = math.floor(365 * n_mesi / 12)
    d, m, y = (int(s) for s in (re.findall(r'\b\d+\b', start)))
    giorni_prec = 0
    for curr_m in range(m)[1:]:
        giorni_prec = giorni_prec + 31 - 3 * (curr_m == 2) + (-1) * (
                curr_m == 4 or curr_m == 6 or curr_m == 9 or curr_m == 11)
    giorni_prec = giorni_prec + 1 * (y == 2020) + 366 * (y == 2021)
    index_start = -54 + giorni_prec + d

    dfi = dfi.loc[index_start: index_start + n_giorni - 1]

    I_vec = np.array(dfi).transpose()

    # Inizializzazione dell'array di output
    K = 1 / 3.6 #percentuale dei timestep usati per interpolare gli infetti, già qui viene fatto uno smoothing
    n_interp_point = max(10,math.floor(n_giorni * K)) #numero punti in cui interpolare gli infetti, circa 
    new_indices = np.linspace(0, I_vec.shape[1] - 1, n_interp_point)
    inf_interp = np.zeros(shape=(I_vec.shape[0], n_interp_point))
    spline_indices = np.linspace(0, I_vec.shape[1] - 1, n_timesteps + 1)
    inf_spline = np.zeros(shape=(I_vec.shape[0], n_timesteps + 1))

    # Interpolazione lineare lungo l'asse N per ogni riga
    for i in range(I_vec.shape[0]):
        inf_interp[i, :] = np.interp(new_indices, np.arange(I_vec.shape[1]), I_vec[i, :])
        #       beta_interp[i, :] = np.interp(new_indices, np.arange(I_vec.shape[1]), I_vec[i, :])
        spline = make_interp_spline(new_indices, inf_interp[i, :])
        inf_spline[i, :] = spline(spline_indices)

    eps = 10 ** (-6)
    inf_spline = inf_spline + eps
    # aggiunta Rt e beta con derivata logaritmica"
    alpha = 1.3
    dt = 1.0 / n_timesteps
    beta_log = np.zeros(shape = (inf_spline.shape[0], inf_spline.shape[1] -1))
    for i in range(inf_spline.shape[1] - 1):
        beta_log[:,i] = np.divide(inf_spline[:, i + 1] - inf_spline[:, i] , n_mesi*dt * inf_spline[:, i]) + alpha
    return inf_spline[:, 0 : -1], beta_log
