import pandas as pd
import numpy as np
import glob

pd.options.display.max_columns = 21


def extract_infects(path, n_timesteps):
    dir_path = path
    #ngiorni = len([entry for entry in os.listdir(dir_path) if os.path.isfile(os.path.join(dir_path, entry))])
    ngiorni = 366
    
    
    nreg = 21
    all_files = sorted(glob.glob(dir_path + "/*.csv"))
    df = pd.read_csv(all_files[0])
    
    reg_list = list(df['denominazione_regione'])
    reg_list.remove('P.A. Trento')
    reg_list.remove('P.A. Bolzano')
    reg_list.insert(len(reg_list),'P.A. Bolzano')
    reg_list.insert(len(reg_list),'P.A. Trento')
    nomi_regioni = dict.fromkeys(reg_list, 0)
    print(nomi_regioni)
    df=df.reindex(columns=reg_list)
    tot_regione = {
    	'Lombardia' :	9950742,
    	'Lazio'	: 5707112,
    	'Campania':	5592175,
    	'Veneto':	4838253,
      'Sicilia' :	4802016,
      'Emilia-Romagna' :	4426929,
    	'Piemonte' :	4240736,
    	'Puglia' :	3900852,
     	'Toscana' : 3651152,
    	'Calabria' : 1841300,
     	'Sardegna' :	1575028,
    	'Liguria' :	1502624,
    	'Marche' :	1480839,
    	'Abruzzo' :	1269860,
    	'Friuli Venezia Giulia'	: 1192191,
    	'Umbria'	: 854137,
    	'Basilicata'	: 536659,
    	'Molise'	: 289840,
    	"Valle d'Aosta"	: 122955,
    	'P.A. Trento' : 542158,
    	'P.A. Bolzano' : 532616}
    tot_regione = pd.Series(tot_regione)
    
    #dfi.loc[0] = pd.Series(np.array(df['totale_positivi']), index=dfi.columns)
    
    
    #giorno = 1
    dfi = pd.DataFrame(data = nomi_regioni, index = [0])
    giorno = 0
    start_vec = [0]; #giorno da cui iniziare a registrare infetti
    for file in (all_files[start_vec[0]: start_vec[0] + ngiorni]):
        df = pd.read_csv(file)
        dfi.loc[giorno] = pd.Series(np.array(df['totale_positivi']), index=dfi.columns)
        # infetti[giorno][:] = (np.array(df['totale_positivi']))
        giorno = giorno + 1
    
    dfi = dfi / tot_regione
    print(dfi.columns)
    dfi=dfi.reindex(columns=reg_list)
    print(dfi.columns)
    dfi = dfi.drop(columns = ['Molise', "Valle d'Aosta"])
    dfi = dfi.loc[0:311]
    
    I_vec = np.array(dfi).transpose()
    new_indices = np.linspace(0, I_vec.shape[1] - 1, n_timesteps)

    # Inizializzazione dell'array di output
    arr_interp = np.zeros((I_vec.shape[0], n_timesteps))
    
    # Interpolazione lineare lungo l'asse N per ogni riga
    for i in range(I_vec.shape[0]):
        arr_interp[i, :] = np.interp(new_indices, np.arange(I_vec.shape[1]), I_vec[i, :])
    return arr_interp
