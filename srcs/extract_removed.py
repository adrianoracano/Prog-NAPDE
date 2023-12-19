import math
import pandas as pd
import numpy as np
import glob
import re
from scipy.interpolate import make_interp_spline

pd.options.display.max_columns = 21


def extract_removed(path, n_timesteps, start, n_mesi):
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

    # dfr.loc[0] = pd.Series(np.array(df['totale_positivi']), index=dfr.columns)

    # giorno = 1
    dfr = pd.DataFrame(data=nomi_regioni, index=[0])
    dfm = pd.DataFrame(data=nomi_regioni, index=[0])
    # dfr.loc[0] = pd.Series(np.array(df['dimessi_guariti']), index=dfr.columns)
    # dfm.loc[0] = pd.Series(np.array(df['deceduti']), index=dfm.columns)
    giorno = 0
    start_vec = [0];  # giorno da cui iniziare a registrare infetti
    for file in (all_files[start_vec[0]: start_vec[0] + ngiorni]):
        df = pd.read_csv(file)
        dfr.loc[giorno] = pd.Series(np.array(df['dimessi_guariti']), index=dfr.columns)
        dfm.loc[giorno] = pd.Series(np.array(df['deceduti']), index=dfr.columns)
        # infetti[giorno][:] = (np.array(df['totale_positivi']))
        giorno = giorno + 1

    dfr = dfr + dfm
    dfr = dfr / tot_regione
    dfr = dfr.reindex(columns=reg_list)
    dfr = dfr.drop(columns=['Molise', "Valle d'Aosta"])

    n_giorni = math.floor(365 * n_mesi / 12)
    d, m, y = (int(s) for s in (re.findall(r'\b\d+\b', start)))
    giorni_prec = 0
    for curr_m in range(m)[1:]:
        giorni_prec = giorni_prec + 31 - 3 * (curr_m == 2) + (-1) * (
                curr_m == 4 or curr_m == 6 or curr_m == 9 or curr_m == 11)
    giorni_prec = giorni_prec + 1 * (y == 2020) + 366 * (y == 2021)
    index_start = -54 + giorni_prec + d;

    dfr = dfr.loc[index_start: index_start + n_giorni - 1]

    R_vec = np.array(dfr).transpose()

    # Inizializzazione dell'array di output
    new_indices = np.linspace(0, R_vec.shape[1] - 1, math.floor(50 * n_mesi / 12))
    rem_interp = np.zeros(shape=(R_vec.shape[0], math.floor(50 * n_mesi / 12)))
    spline_indices = np.linspace(0, R_vec.shape[1] - 1, n_timesteps)
    rem_spline = np.zeros(shape=(R_vec.shape[0], n_timesteps))

    # Interpolazione lineare lungo l'asse N per ogni riga
    for i in range(R_vec.shape[0]):
        rem_interp[i, :] = np.interp(new_indices, np.arange(R_vec.shape[1]), R_vec[i, :])
        #       beta_interp[i, :] = np.interp(new_indices, np.arange(R_vec.shape[1]), R_vec[i, :])
        spline = make_interp_spline(new_indices, rem_interp[i, :])
        rem_spline[i, :] = spline(spline_indices)

    # aggiunta Rt e beta con derivata logaritmica"
    # alpha = 1.3
    # mylog = np.vectorize(math.log)
    # dt = n_mesi * 30 / n_timesteps
    # R0_log = (mylog(rem_spline[:, 1]) - mylog(rem_spline[:, 0])) / (dt * alpha) + 1
    # beta0_log = R0_log * alpha;
    return rem_spline
