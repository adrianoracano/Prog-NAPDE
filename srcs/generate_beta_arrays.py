from srcs import extract_infects as exi
from srcs import extract_removed as exr
import math
import numpy as np
from scipy.interpolate import make_interp_spline
from utilities import SirHelperFunctions as shr
import converti_ngiorni_data as cgd
from matplotlib import pyplot as pltS
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

#funzione per generare dataset infetti ===========================

#path_i = lo stesso di extract_infects
#K = numero di sample da generare, se sono troppi dà errore
#n_giorni = lunghezza di un intervallo in giorni
#overlap = overlap tra i periodi
#start_date = data di inizio
#n_timesteps = timesteps di un intervallo (facoltativo)
#regions = ['nome della regione'], in caso di più regioni torna un array (n_regioni * n_intervalli) * n_timesteps
#dove le prime n_regioni colonne sono gli intervalli per quella regioni dalla data 1, le seconde dalla data 2 ecc..

def generate_beta_arrays(path_i, K, n_giorni, overlap, start_date  = '15/03/2020', n_timesteps = None, regions = reg_list):
    n_mesi = n_giorni / 30
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
    
    #vecchia parte con max_months commentata
    """ start = cgd.data_a_ngiorni(start_date)
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
   
    #vecchia parte con max_months commentata
    """ 
    #calcolo del numero di intervalli
    used_days = n_giorni
    count = 1
    while used_days <= max_days - n_giorni + overlap:
        used_days += n_giorni - overlap
        count += 1
    print(str(count) + " intervals can be generated") """

    if n_timesteps == None:
        n_timesteps = max(n_giorni*2, 100)
    
    infetti, beta_log = exi.extract_infects(path_i, n_timesteps, start_date, n_mesi, regions)
    rimossi =  exr.extract_removed(path_i, n_timesteps, start_date, n_mesi, regions)
    start_new = cgd.ngiorni_a_data(cgd.data_a_ngiorni(start_date) + n_giorni - overlap)
    date = [start_new]
    #vecchia parte con max_months commentata
    """ 
    for i in range(count - 1):
        start_new = cgd.ngiorni_a_data(cgd.data_a_ngiorni(start_new) + n_giorni - overlap)
        infetti_new, beta_log_new = exi.extract_infects(path_i, n_timesteps, start_new, n_mesi, regions)
        infetti = np.concatenate((infetti, infetti_new), axis = 0)
        beta_log = np.concatenate((beta_log, beta_log_new), axis = 0)
        rimossi_new =  exr.extract_removed(path_i, n_timesteps, start_new, n_mesi, regions)
        rimossi = np.concatenate((rimossi, rimossi_new), axis = 0) """

    for i in range(K - 1):
        start_new = cgd.ngiorni_a_data(cgd.data_a_ngiorni(start_new) + n_giorni - overlap)
        print(start_new)
        date.append(start_new)
        infetti_new, beta_log_new = exi.extract_infects(path_i, n_timesteps, start_new, n_mesi, regions)
        infetti = np.concatenate((infetti, infetti_new), axis = 0)
        beta_log = np.concatenate((beta_log, beta_log_new), axis = 0)
        rimossi_new =  exr.extract_removed(path_i, n_timesteps, start_new, n_mesi, regions)
        rimossi = np.concatenate((rimossi, rimossi_new), axis = 0)

    #smoothing infetti ===============================================
    """ 
    smooth = False
    if smooth :
        # Inizializzazione dell'array di output
        I_vec = infetti
        K_I = 1 / 2 #percentuale dei timestep usati per interpolare gli infetti, si moltiplica a quello in exi
        n_interp_point = max(10,math.floor(n_giorni * K_I)) #numero punti in cui interpolare gli infetti
        new_indices = np.linspace(0, I_vec.shape[1] - 1, n_interp_point)
        inf_interp = np.zeros(shape=(I_vec.shape[0], n_interp_point))
        spline_indices = np.linspace(0, I_vec.shape[1] - 1, n_timesteps + 1)
        inf_spline = np.zeros(shape=(I_vec.shape[0], n_timesteps + 1))
 """
    # beta sintetico =================================================
    use_positive = True #per tagliare beta_log negativo
    if use_positive:
        beta_log = beta_log.clip(min = 0)
    up_bound = 20 #per tagliare beta sopra una soglia
    use_up_bound = True
    if use_up_bound:
        beta_log = beta_log.clip(max = up_bound)
    K_b = 1 / 3 # percentuale di timestep per interpolare beta, regola lo smoothing di beta
    n_interp_points = math.floor((n_timesteps) * K_b)
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
    noise = noise + 0.1* amp * vec
    #rumore 2(frequenza bassa, ampiezza alta)
    vec2 = sin(omega/5 * t)
    vec2 = vec2.reshape(1, beta_spline.shape[1])
    np.tile(vec2, beta_spline.shape[1])
    noise = noise + 0.2 * amp * vec2
    beta_spline = beta_spline + noise

    #rimozione parti negative e resmoothing
    beta_spline.clip(min = 0) 
    K_b2 = 1 / 5 # per smoothare la parte = 0
    n_interp_points = math.floor((n_timesteps) * K_b2)
    beta_spline_interp = np.zeros(shape=(beta_spline.shape[0], n_interp_points))
    new_indices = np.linspace(0, beta_log.shape[1], n_interp_points)
    for i in range(beta_log.shape[0]):
        beta_spline_interp[i, :] = np.interp(new_indices, np.arange(beta_spline.shape[1]), beta_spline[i, :])
        spline = make_interp_spline(new_indices, beta_spline_interp[i, :])
        beta_spline[i, :] = spline(spline_indices)

    #generazione degli infetti tramite il SIR=========================
    R0 = rimossi[:,0]
    I0 = infetti[:,0]
    S0 = 1 - I0 - R0
    sir0 = (S0, I0, R0)
    alpha = 1.3
    infetti_SIR = shr.compute_I([beta_spline], n_mesi, alpha, sir0)
    infetti_SIR = infetti_SIR[0]

    """ 
    #select K_val samples randomly
    infetti_SIR_val = np.zeros(shape = (K_val, n_timesteps))
    beta_spline_val = np.zeros(shape = (K_val, n_timesteps))
    for i in range(K_val):
        random_id = random.randint(0, (infetti_SIR.shape)[0] - 1)
        infetti_SIR_val[i, :] = infetti_SIR_val[random_id, :]
        infetti_SIR = np.delete(infetti_SIR, random_id, axis = 0)
        print(infetti_SIR.shape)
        beta_spline_val[i, :] = beta_spline_val[random_id, :]
        beta_spline = np.delete(beta_spline, random_id, axis = 0)
      """

    return beta_spline, infetti_SIR, date #, beta_spline_val, infetti_SIR_val



