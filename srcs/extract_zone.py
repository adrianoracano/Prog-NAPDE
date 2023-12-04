import pandas as pd
import numpy as np
import glob
import re

pd.options.display.max_columns = 21

def extract_zones(n_timesteps, start):

    n_giorni = 365

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
     'Molise',
     'Piemonte',
     'Puglia',
     'Sardegna',
     'Sicilia',
     'Toscana',
     'Umbria',
     "Valle d'Aosta",
     'Veneto',
     'P.A. Bolzano',
     'P.A. Trento']

    nomi_regioni = dict.fromkeys(reg_list, 0)
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
    
    
    #creazione del dataset di zone
    #inidici del dataset
    index = []
    anno = 2020
    mesi = [' Gennaio ', ' Febbraio ',  ' Marzo ',' Aprile ',' Maggio ', ' Giugno ', ' Luglio ',  ' Agosto ', ' Settembre ', ' Ottobre ', ' Novembre ', ' Dicembre ']
    for n_anni in range(3):
      if(anno == 2020):
        bisestile = 1
      else:
        bisestile = 0
      for mese in mesi:
        shift = -1 * ( mese == ' Aprile ' or mese == ' Giugno ' or mese ==' Settembre ' or mese == ' Novembre ')  - (3-bisestile) * (mese == ' Febbraio ')
        for giorno in range(31 + shift):
          giorno = giorno + 1
          index.append(str(giorno) + mese + str(anno))
      anno = anno + 1
    
    index = index[54:]#inizio dal 24 feb 2020
    
    r, a, g, b = [0.2, 0.46, 0.73, 1]
    
    dsz = pd.DataFrame(data = nomi_regioni, index = index)
    init = np.ones(shape = ( len(index), len(dsz.columns))) * b
    dsz.loc[:] = init

    #creazione tupla di regioni per riempire il dataset
    Abruzzo = 'Abruzzo'
    Basilicata = 'Basilicata'
    Calabria = 'Calabria'
    Campania = 'Campania'
    EmiliaRomagna = 'Emilia-Romagna'
    FriuliVeneziaGiulia = 'Friuli Venezia Giulia'
    Lazio = 'Lazio'
    Liguria = 'Liguria'
    Lombardia = 'Lombardia'
    Marche = 'Marche'
    Molise = 'Molise'
    PABolzano = 'P.A. Bolzano'
    PATrento = 'P.A. Trento'
    Piemonte = 'Piemonte'
    Puglia = 'Puglia'
    Sardegna = 'Sardegna'
    Sicilia = 'Sicilia'
    Toscana = 'Toscana'
    Umbria = 'Umbria'
    ValledAosta = "Valle d'Aosta"
    Veneto = 'Veneto'

    tupla_reg = (
    Abruzzo, Basilicata, Calabria, Campania, EmiliaRomagna, FriuliVeneziaGiulia, Lazio, Liguria, Lombardia, Marche,
    Molise, PABolzano, PATrento, Piemonte, Puglia, Sardegna, Sicilia, Toscana, Umbria, ValledAosta, Veneto)

    #riempimento del dataset di zone , si poteva anche usare la funzione:
    def riempi(dsz, data1, data2, regioni, colore):
      if not (len(regioni) == 0):
        #regioni è una tupla : (regione1, regione2, ...)
        dsz.loc[data1 : data2, regioni] = np.ones(shape = np.array(dsz.loc[data1 : data2, regioni]).shape) * colore
      else:
        dsz.loc[data1 : data2] = np.ones(shape = np.array(dsz.loc[data1 : data2]).shape) * colore
    
    dsz['Lombardia'].loc['24 Febbraio 2020' : '8 Marzo 2020'] = np.ones(shape = np.array(dsz['Lombardia'].loc['24 Febbraio 2020' : '8 Marzo 2020']).shape) * r
    
    dsz.loc['1 Marzo 2020' : '8 Marzo 2020', ('Veneto','Emilia-Romagna', 'Marche')] = np.ones(shape = np.array(dsz.loc['1 Marzo 2020' : '8 Marzo 2020', ('Veneto','Emilia-Romagna', 'Marche')]).shape) * r
    
    dsz.loc['9 Marzo 2020' : '4 Maggio 2020'] = np.ones(shape = np.array(dsz.loc['9 Marzo 2020' : '4 Maggio 2020']).shape) * r
    
    dsz.loc['5 Maggio 2020' : '17 Maggio 2020'] = np.ones(shape = np.array(dsz.loc['5 Maggio 2020' : '17 Maggio 2020']).shape) * a
    
    dsz.loc['18 Maggio 2020' : '10 Giugno 2020'] = np.ones(shape = np.array(dsz.loc['18 Maggio 2020' : '10 Giugno 2020']).shape) * g
    
    dsz.loc['10 Giugno 2020' : '12 Ottobre 2020'] = np.ones(shape = np.array(dsz.loc['10 Giugno 2020' : '12 Ottobre 2020']).shape) * b
    
    dsz.loc['10 Giugno 2020' : '12 Ottobre 2020'] = np.ones(shape = np.array(dsz.loc['10 Giugno 2020' : '12 Ottobre 2020']).shape) * b
    
    data1 = '13 Ottobre 2020'
    data2 = '23 Ottobre 2020'
    dsz.loc[data1 : data2] = np.ones(shape = np.array(dsz.loc[data1 : data2]).shape) * g
    
    data1 = '23 Ottobre 2020'
    data2 = '2 Novembre 2020'
    dsz.loc[data1 : data2] = np.ones(shape = np.array(dsz.loc[data1 : data2]).shape) * a
    
    data1 = '3 Novembre 2020'
    data2 = '2 Dicembre 2020'
    dsz.loc[data1 : data2] = np.ones(shape = np.array(dsz.loc[data1 : data2]).shape) * r
    
    data1 = '3 Dicembre 2020'
    data2 = '23 Dicembre 2020'
    dsz.loc[data1 : data2] = np.ones(shape = np.array(dsz.loc[data1 : data2]).shape) * a
    
    data1 = '24 Dicembre 2020'
    data2 = '27 Dicembre 2020'
    dsz.loc[data1 : data2] = np.ones(shape = np.array(dsz.loc[data1 : data2]).shape) * r
    
    data1 = '28 Dicembre 2020'
    data2 = '30 Dicembre 2020'
    dsz.loc[data1 : data2] = np.ones(shape = np.array(dsz.loc[data1 : data2]).shape) * a
    
    data1 = '31 Dicembre 2020'
    data2 = '6 Gennaio 2021'
    dsz.loc[data1 : data2] = np.ones(shape = np.array(dsz.loc[data1 : data2]).shape) * r
    
    data1 = '7 Gennaio 2021'
    data2 = '14 Gennaio 2021'
    dsz.loc[data1 : data2] = np.ones(shape = np.array(dsz.loc[data1 : data2]).shape) * a
    
    data1 = '7 Gennaio 2021'
    data2 = '23 Febbraio 2021'
    dsz.loc[data1 : data2] = np.ones(shape = np.array(dsz.loc[data1 : data2]).shape) * r

    riempi(dsz, '1 Marzo 2021', '7 Marzo 2021',
           ('Calabria', 'Friuli Venezia Giulia', 'Lazio', 'Liguria', 'Puglia', 'Sicilia', "Valle d'Aosta", 'Veneto'), g)
    riempi(dsz, '1 Marzo 2021', '7 Marzo 2021', ('Basilicata', 'Molise'), r)
    # riempi(dsz, '1 Marzo 2021', '7 Marzo 2021', 'Sardegna', b)
    riempi(dsz, '1 Marzo 2021', '7 Marzo 2021', (
    'Abruzzo', 'Campania', 'Emilia-Romagna', 'Lombardia', 'Marche', 'Piemonte', 'Toscana', 'P.A. Bolzano',
    'P.A. Trento', 'Umbria'), a)

    riempi(dsz, '8 Marzo 2021', '14 Marzo 2021', (Calabria, Lazio, Liguria, Puglia, Sicilia, ValledAosta), g)
    riempi(dsz, '8 Marzo 2021', '14 Marzo 2021', (
    Abruzzo, EmiliaRomagna, FriuliVeneziaGiulia, Lombardia, Marche, Piemonte, Toscana, PABolzano, PATrento, Umbria,
    Veneto), a)
    riempi(dsz, '8 Marzo 2021', '14 Marzo 2021', (Basilicata, Campania, Molise), r)
    # riempi(dsz, '8 Marzo 2021', '14 Marzo 2021', 'Sardegna', b)

    riempi(dsz, '15 Marzo 2021', '21 Marzo 2021',
           (Abruzzo, Basilicata, Calabria, Liguria, PABolzano, Sicilia, Toscana, Umbria, ValledAosta), a)
    riempi(dsz, '15 Marzo 2021', '21 Marzo 2021', (
    Abruzzo, EmiliaRomagna, FriuliVeneziaGiulia, Lombardia, Marche, Piemonte, Toscana, PABolzano, PATrento, Umbria,
    Veneto), a)
    riempi(dsz, '15 Marzo 2021', '21 Marzo 2021', (
    Campania, EmiliaRomagna, FriuliVeneziaGiulia, Lazio, Lombardia, Marche, Molise, PATrento, Piemonte, Puglia, Veneto),
           r)
    # riempi(dsz, '15 Marzo 2021', '21 Marzo 2021', 'Sardegna', b)

    riempi(dsz, '22 Marzo 2021', '28 Marzo 2021', tupla_reg, a)
    riempi(dsz, '22 Marzo 2021', '28 Marzo 2021',
           (Campania, EmiliaRomagna, FriuliVeneziaGiulia, Lazio, Lombardia, Marche, Piemonte, PATrento, Puglia, Veneto),
           r)

    riempi(dsz, '29 Marzo 2021', '5 Aprile 2021', tupla_reg, a)
    riempi(dsz, '29 Marzo 2021', '5 Aprile 2021', (
    Calabria, Campania, EmiliaRomagna, FriuliVeneziaGiulia, Lazio, Lombardia, Marche, Piemonte, PATrento, Puglia,
    Toscana, ValledAosta, Veneto), r)

    riempi(dsz, '6 Aprile 2021', '11 Aprile 2021', tupla_reg, a)
    riempi(dsz, '6 Aprile 2021', '11 Aprile 2021',
           (Calabria, Campania, EmiliaRomagna, FriuliVeneziaGiulia, Lombardia, Piemonte, Puglia, Toscana, ValledAosta),
           r)

    riempi(dsz, '12 Aprile 2021', '18 Aprile 2021', tupla_reg, a)
    riempi(dsz, '12 Aprile 2021', '18 Aprile 2021', (Campania, Puglia, Sardegna, ValledAosta), r)

    riempi(dsz, '19 Aprile 2021', '25 Aprile 2021', tupla_reg, a)
    riempi(dsz, '19 Aprile 2021', '25 Aprile 2021', (Puglia, Sardegna, ValledAosta), r)

    riempi(dsz, '26 Aprile 2021', '2 Maggio 2021', (Sardegna), r)
    riempi(dsz, '26 Aprile 2021', '2 Maggio 2021', (Basilicata, Calabria, Puglia, Sicilia, ValledAosta), a)
    riempi(dsz, '26 Aprile 2021', '2 Maggio 2021', (
    Abruzzo, Campania, EmiliaRomagna, FriuliVeneziaGiulia, Lazio, Liguria, Lombardia, Marche, Molise, Piemonte,
    PABolzano, PATrento, Toscana, Umbria, Veneto), g)

    riempi(dsz, '3 Maggio 2021', '9 Maggio 2021', ("Valle d'Aosta"), r)
    riempi(dsz, '3 Maggio 2021', '9 Maggio 2021', (Basilicata, Calabria, Puglia, Sicilia, Sardegna), a)
    riempi(dsz, '3 Maggio 2021', '9 Maggio 2021', (
    Abruzzo, Campania, EmiliaRomagna, FriuliVeneziaGiulia, Lazio, Liguria, Lombardia, Marche, Molise, Piemonte,
    PABolzano, PATrento, Toscana, Umbria, Veneto), g)

    riempi(dsz, '10 Maggio 2021', '16 Maggio 2021', (Sardegna, Sicilia, ValledAosta), a)
    riempi(dsz, '10 Maggio 2021', '16 Maggio 2021', (
    Abruzzo, Basilicata, Calabria, Campania, EmiliaRomagna, FriuliVeneziaGiulia, Lazio, Liguria, Lombardia, Marche,
    Molise, Piemonte, PABolzano, PATrento, Puglia, Toscana, Umbria, Veneto), g)

    riempi(dsz, '17 Maggio 2021', '23 Maggio 2021', tupla_reg, g)
    riempi(dsz, '17 Maggio 2021', '23 Maggio 2021', (ValledAosta), a)

    riempi(dsz, '24 Maggio 2021', '30 Maggio 2021', tupla_reg, g)

    riempi(dsz, '31 Maggio 2021', '6 Giugno 2021', tupla_reg, g)
    riempi(dsz, '31 Maggio 2021', '6 Giugno 2021', (FriuliVeneziaGiulia, Molise, Sardegna), b)

    riempi(dsz, '7 Giugno 2021', '13 Giugno 2021',
           (Basilicata, Calabria, Campania, Marche, PABolzano, Sicilia, Toscana, ValledAosta), g)
    riempi(dsz, '7 Giugno 2021', '13 Giugno 2021', (
    Abruzzo, EmiliaRomagna, FriuliVeneziaGiulia, Lazio, Liguria, Lombardia, Molise, Piemonte, PATrento, Puglia,
    Sardegna, Umbria, Veneto), b)

    riempi(dsz, '14 Giugno 2021', '20 Giugno 2021',
           (Basilicata, Calabria, Campania, Marche, PABolzano, Sicilia, Toscana, ValledAosta), g)
    riempi(dsz, '14 Giugno 2021', '20 Giugno 2021', (
    Abruzzo, EmiliaRomagna, FriuliVeneziaGiulia, Lazio, Liguria, Lombardia, Molise, Piemonte, PATrento, Puglia,
    Sardegna, Umbria, Veneto), b)

    riempi(dsz, '21 Giugno 2021', '27 Giugno 2021', tupla_reg, b)
    riempi(dsz, '21 Giugno 2021', '27 Giugno 2021', (ValledAosta), g)
    
    dsz=dsz.reindex(columns=reg_list)
    
    dsz = dsz.drop(columns = ['Molise', "Valle d'Aosta"])

    n_giorni = 365
    d, m, y = (int(s) for s in (re.findall(r'\b\d+\b', start)))
    giorni_prec = 0
    for curr_m in range(m)[1:]:
        giorni_prec = giorni_prec + 31 - 3 * (curr_m == 2) + (-1) * (
                curr_m == 4 or curr_m == 6 or curr_m == 9 or curr_m == 11)
    giorni_prec = giorni_prec + 1 * (y == 2020)
    index_start = giorni_prec + d;

    dsz = dsz.loc[index[index_start : index_start + n_giorni]]
    
    Z_vec = np.array(dsz).transpose()
    arr_interp = np.zeros((Z_vec.shape[0], n_timesteps))
    new_indices = np.linspace(0, Z_vec.shape[1] - 1, n_timesteps)

    # Interpolazione lineare lungo l'asse N per ogni riga
    for i in range(Z_vec.shape[0]):
        arr_interp[i, :] = np.interp(new_indices, np.arange(Z_vec.shape[1]), Z_vec[i, :])
    return arr_interp
