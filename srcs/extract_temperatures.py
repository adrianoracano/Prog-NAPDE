import pandas as pd
import numpy as np
import os
import re
import math
from scipy.interpolate import make_interp_spline


def extract_temperatures(path, n_timesteps, start, n_mesi, file_prefix =  ['Milano']):

    # Specify the file path
    #              
    file_month = ["Febbraio", "Marzo", "Aprile", "Maggio", "Giugno", "Luglio", "Agosto", "Settembre", "Ottobre",
                  "Novembre", "Dicembre"]
    
    # Specify the base directory relative to the script location
    
    # Create an array of zeros with shape (19, 335)
    tmedia_val = np.zeros((len(file_prefix), 335 + 365))
    
    # Initialize the last filled index for each location
    last_filled_index = np.zeros(len(file_prefix), dtype=int)
    
    # Load and append data from additional CSV files for each location and month
    for anno in ['2020', '2021']:
        if anno == '2020':
            months_list = file_month
        else:
            months_list = ['Gennaio'] + file_month
        for month_idx, month in enumerate(months_list):
            for location_idx, prefix in enumerate(file_prefix):

                try:
                    file_path = os.path.join(path, prefix, f"{prefix}-" + anno + f"-{month}.csv")
                    df_additional = pd.read_csv(file_path, delimiter=';', encoding='utf-8')
                    tmedia_additional = df_additional["TMEDIA °C"].values
                    #print(f"Loaded data for {prefix} in {month} - Shape: {tmedia_additional.shape}, Values: {tmedia_additional}")

                    # Get the last filled index for the current location
                    start_index = last_filled_index[location_idx]

                    # Update the last filled index for the next iteration
                    last_filled_index[location_idx] = start_index + len(tmedia_additional)

                    # Fill the values in the array
                    tmedia_val[location_idx, start_index:last_filled_index[location_idx]] = tmedia_additional

                except FileNotFoundError as e:
                    print(f"Error: File not found at {file_path}")
                    raise e

    n_giorni = math.floor(365 * n_mesi / 12)
    d, m, y = (int(s) for s in (re.findall(r'\b\d+\b', start)))
    giorni_prec = 0
    for curr_m in range(m)[1:]:
        giorni_prec = giorni_prec + 31 - 3 * (curr_m == 2) + (-1) * (
                curr_m == 4 or curr_m == 6 or curr_m == 9 or curr_m == 11)
    giorni_prec = giorni_prec + 1 * (y == 2020) + 366 * (y == 2021)
    index_start = -54 + giorni_prec + d;

    tmedia_val= tmedia_val[:, index_start : index_start + n_giorni]

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

    tmedia_val = replace_nan_with_previous(tmedia_val)
    return tmedia_val

    new_indices = np.linspace(0, tmedia_val.shape[1] - 1, math.floor(50 * n_mesi / 12))

    # Inizializzazione dell'array di output
    arr_interp = np.zeros((tmedia_val.shape[0], math.floor(50 * n_mesi / 12)))
    spline_indices = np.linspace(0, tmedia_val.shape[1] - 1, n_timesteps)
    T_spline = np.zeros((tmedia_val.shape[0], n_timesteps))
    # Interpolazione lineare lungo l'asse N per ogni riga
    for i in range(tmedia_val.shape[0]):
        arr_interp[i, :] = np.interp(new_indices, np.arange(tmedia_val.shape[1]), tmedia_val[i, :])
        spline = make_interp_spline(new_indices, arr_interp[i, :])
        T_spline[i, :] = spline(spline_indices)
    return T_spline



