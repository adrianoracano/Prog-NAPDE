from srcs import extract_temperatures as ext
import numpy as np
import math
import re
import converti_ngiorni_data as cgd

reg_list = ['Milano']

def generate_temp_arrays(path_t, n_giorni, max_months, overlap, start_date  = '15/03/2020', n_timesteps = None, region = reg_list):
    n_mesi = n_giorni / 30
    
    start = cgd.data_a_ngiorni(start_date)
    max_days = math.floor(max_months * 30)
    end = start + max_days
    end_date = cgd.ngiorni_a_data(end)
    end_d, end_m, end_y = (int(s) for s in (re.findall(r'\b\d+\b', end_date)))
    if end_y > 2021 or (end_y == 2021 and end_m>9) or (end_y == 2021 and end_m == 9 and end_d > 15):
        raise("Too many days")
    if overlap > n_giorni or n_giorni > max_days or overlap > max_days:
        raise("Incompatible sizes")
    
    #calcolo del numero di intervalli
    used_days = n_giorni
    count = 1
    while used_days <= max_days - n_giorni + overlap:
        used_days += n_giorni - overlap
        count += 1
    print(str(count) + " intervals can be generated")

    if n_timesteps == None:
        n_timesteps = n_giorni*2


    temp = ext.extract_temperatures(path_t, n_timesteps, start_date, n_mesi, region)
    start_new = cgd.ngiorni_a_data(cgd.data_a_ngiorni(start_date) + n_giorni - overlap)
    for i in range(count - 1):
        start_new = cgd.ngiorni_a_data(cgd.data_a_ngiorni(start_new) + n_giorni - overlap)
        temp_new = ext.extract_temperatures(path_t, n_timesteps, start_new, n_mesi)
        temp = np.concatenate((temp, temp_new), axis = 0)
        
    
    return temp

    