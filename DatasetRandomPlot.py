import tensorflow as tf
import os
import numpy as np
import MyTemperatureGenerator as tg
import MyDatasetGenerator as dsg
from utilities import HerRungeKutta as hrk
import argparse
import sys
import pickle
from matplotlib import pyplot as plt
from utilities import SirHelperFunctions as shf
import random
import math
tf.keras.backend.set_floatx('float64')

data_dict = {}
with open('data.txt', 'r') as file:
    for line in file:
        line = line.strip()  # Remove leading/trailing whitespace
        if line:
            field, value = line.split(':')
            data_dict[field.strip()] = value.strip()

nome_file_temp = 'datasets/'+data_dict['dataset']
try:
    with open(nome_file_temp, 'rb') as file:
        dataset, K, val_set, K_val, test_set, K_test = pickle.load(file)  # viene caricato il  dataset
    print('dataset', nome_file_temp, 'loaded...\n')
except FileNotFoundError:
    print('file',nome_file_temp,'not found...\n')
    sys.exit()


N = int(data_dict['N'])
index1 = random.randint(1,len(dataset[0,:,0]))
index2 = random.randint(1,len(dataset[0,:,0]))
t = np.linspace(0., 1, N)
fig, (ax1, ax2, ax3) = plt.subplots(3)
fig.suptitle('Temperature, beta, betaeq')
ax1.plot(t, dataset[0, index1, ])
ax1.plot(t, dataset[0, index2, ])
ax1.legend(["T1(t)", "T2(t)"])
ax2.plot(t, dataset[2, index1, ])
ax2.plot(t, dataset[2, index2, ])
ax2.legend(["betaeq1(t)", "betaeq2(t)"])
ax3.plot(t, dataset[1, index1, ])
ax3.plot(t, dataset[1, index2, ])
ax3.legend(["beta1(t)", "beta2(t)"])
plt.show()

