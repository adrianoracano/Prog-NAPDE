from srcs import extract_temperatures as ext
import numpy as np
import math
import re
import converti_ngiorni_data as cgd

reg_list = ["Chieti", "Matera", "Cosenza", "Napoli", "Bologna", "Trieste", "Roma",
                    "Genova", "Milano", "Ancona", "Torino", "Bari", "Cagliari","Palermo", 
                    "Firenze", "Perugia", "Verona", "Bolzano", "Trento"]

def generate_temp_arrays(path_t, K, n_giorni, overlap, start_date  = '15/03/2020', n_timesteps = None, regions = reg_list):
    n_mesi = n_giorni / 30

    reg_list = ["Chieti", "Matera", "Cosenza", "Napoli", "Bologna", "Trieste", "Roma",
                    "Genova", "Milano", "Ancona", "Torino", "Bari", "Cagliari","Palermo", 
                    "Firenze", "Perugia", "Verona", "Bolzano", "Trento"]

    """
    start = cgd.data_a_ngiorni(start_date)
    max_days = math.floor(max_months * 30)
    end = start + max_days
    end_date = cgd.ngiorni_a_data(end)
    end_d, end_m, end_y = (int(s) for s in (re.findall(r'\b\d+\b', end_date)))
    if end_y > 2021 or (end_y == 2021 and end_m>9) or (end_y == 2021 and end_m == 9 and end_d > 15):
        raise("Too many days")
    if overlap > n_giorni or n_giorni > max_days or overlap > max_days:
        raise("Incompatible sizes") """
    
    start = cgd.data_a_ngiorni(start_date)
    end = start + (K) * (n_giorni - overlap) 
    end_date = cgd.ngiorni_a_data(end)
    if end > cgd.data_a_ngiorni("15/09/2021"):
        raise ValueError('Too many intervals')
    
    """ #calcolo del numero di intervalli
    used_days = n_giorni
    count = 1
    while used_days <= max_days - n_giorni + overlap:
        used_days += n_giorni - overlap
        count += 1
    print(str(count) + " intervals can be generated") """

    if n_timesteps == None:
        n_timesteps = max(n_giorni*2, 100)


    temp = ext.extract_temperatures(path_t, n_timesteps, start_date, n_mesi, regions)
    start_new = cgd.ngiorni_a_data(cgd.data_a_ngiorni(start_date) + n_giorni - overlap)
    for i in range(K - 1):
        start_new = cgd.ngiorni_a_data(cgd.data_a_ngiorni(start_new) + n_giorni - overlap)
        temp_new = ext.extract_temperatures(path_t, n_timesteps, start_new, n_mesi, regions)
        temp = np.concatenate((temp, temp_new), axis = 0)
    return temp

def generate_temp_arrays_2(path_t, K, n_giorni, overlap, start_date  = '15/03/2020', n_timesteps_per_interval = None, regions = reg_list):
    n_mesi = n_giorni / 30

    reg_list = ["Chieti", "Matera", "Cosenza", "Napoli", "Bologna", "Trieste", "Roma",
                    "Genova", "Milano", "Ancona", "Torino", "Bari", "Cagliari","Palermo", 
                    "Firenze", "Perugia", "Verona", "Bolzano", "Trento"]

    """
    start = cgd.data_a_ngiorni(start_date)
    max_days = math.floor(max_months * 30)
    end = start + max_days
    end_date = cgd.ngiorni_a_data(end)
    end_d, end_m, end_y = (int(s) for s in (re.findall(r'\b\d+\b', end_date)))
    if end_y > 2021 or (end_y == 2021 and end_m>9) or (end_y == 2021 and end_m == 9 and end_d > 15):
        raise("Too many days")
    if overlap > n_giorni or n_giorni > max_days or overlap > max_days:
        raise("Incompatible sizes") """
    
    start = cgd.data_a_ngiorni(start_date)
    end = start + (K) * (n_giorni - overlap) 
    end_date = cgd.ngiorni_a_data(end)
    if end > cgd.data_a_ngiorni("15/09/2021"):
        raise ValueError('Too many intervals')
    
    """ #calcolo del numero di intervalli
    used_days = n_giorni
    count = 1
    while used_days <= max_days - n_giorni + overlap:
        used_days += n_giorni - overlap
        count += 1
    print(str(count) + " intervals can be generated") """

    if n_timesteps_per_interval == None:
        n_timesteps_per_interval = max((n_giorni - overlap) * 2, 100)

    n_timesteps_per_overlap = math.floor(n_timesteps_per_interval * overlap / n_giorni)
    n_timesteps = (n_timesteps_per_interval - n_timesteps_per_overlap) * K + n_timesteps_per_overlap
    n_mesi_tot = ((n_giorni - overlap) * K + overlap) * 12 / 365
    temp_full = ext.extract_temperatures(path_t, n_timesteps, start_date, n_mesi_tot, regions)

    #first iteration outside the loop
    start = 0
    end = n_timesteps_per_interval
    temp = temp_full[0, start : end].reshape(1,n_timesteps_per_interval)
    start = end - n_timesteps_per_overlap
    end = start + n_timesteps_per_interval 

    #first iteration outside the loop
    start_new = cgd.ngiorni_a_data(cgd.data_a_ngiorni(start_date) + n_giorni - overlap)
    date = [start_new]

    for i in range(K - 1):            
        start_new = cgd.ngiorni_a_data(cgd.data_a_ngiorni(start_new) + n_giorni - overlap)
        print(start_new)
        date.append(start_new)
        temp = np.concatenate((temp, temp_full[0, start : end].reshape(1,n_timesteps_per_interval)), axis = 0)
        start = end - n_timesteps_per_overlap
        end = start + n_timesteps_per_interval 

    return temp

    