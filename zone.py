import pandas as pd
import tensorflow as tf
import numpy as np
import os
import glob
import matplotlib.pyplot as plt
import tensorflow as tf

pd.options.display.max_columns = 21


dir_path = 'COVID-19/dati-regioni/'#bisogna scaricare la repo del dpc e metterla in questa
#ngiorni = len([entry for entry in os.listdir(dir_path) if os.path.isfile(os.path.join(dir_path, entry))])
ngiorni = 366


nreg = 21
all_files = sorted(glob.glob(dir_path + "*.csv"))
df = pd.read_csv(all_files[0])

reg_list = list(df['denominazione_regione'])
reg_list.remove('P.A. Trento')
reg_list.remove('P.A. Bolzano')
reg_list.insert(len(reg_list),'P.A. Bolzano')
reg_list.insert(len(reg_list),'P.A. Trento')
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


#riempimento del dataset di zone , si poteva anche usare la funzione:
# def riempi(dsz, data1, data2, regioni, colore):
#   if len(regioni) == 0:
#     #regioni è una tupla : (regione1, regione2, ...)
#     dsz.loc[data1 : data2, regioni] = np.ones(shape = np.array(dsz.loc[data1 : data2, regioni]).shape) * colore
#   else:
#     dsz.loc[data1 : data2] = np.ones(shape = np.array(dsz.loc[data1 : data2]).shape) * colore

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



dsz=dsz.reindex(columns=reg_list)


dsz = dsz.drop(columns = ['Molise', "Valle d'Aosta"])
dsz = dsz.loc['24 Febbraio 2020' : '31 Dicembre 2020']

Z_vec = np.array(dsz).transpose()
print(Z_vec.shape)

