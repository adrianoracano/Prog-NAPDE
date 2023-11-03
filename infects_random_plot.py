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
import MyDatasetGenerator as dg
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



from utilities import MyRK4

a = float(data_dict['alpha'])
S0 = float(data_dict['S0'])
I0 = float(data_dict['I0'])
R0 = float(data_dict['R0'])
t_max = float(data_dict['t_max'])
alpha = float(data_dict['alpha'])
S_inf = float(data_dict['S_inf'])
TOT = S0 + I0 + R0  # popolazione totale
sir_0 = np.array([S0, I0, R0])
N = int(data_dict['N'])
t = np.linspace(0., 1, N)
I = np.zeros(shape = (N,1))  # da usare per gli infetti
tau = 0.5
def f(beta, betaeq): # è la funzione che regola beta:   beta(t)' = f(beta(t), T(t))
    return (1/tau)*(betaeq - beta)*t_max
data = {'N' : N,
        't_max' : 1,
        'beta0' : np.array([float(data_dict['beta0'])]),
        'f' : f}
b_ref = alpha*math.log(S0/S_inf)/(1-S_inf)

for mese in range(60,70,1):
    tau_noise = 25
    mese = mese/10
    ampiezza_noise = 1.5
    # mese = 5
    ampiezza = 1
    altezza = 1
    Tamp = 25*ampiezza
    Tmean = 25*0.93 + altezza
    def Tcos(t):
        return altezza + ampiezza*(math.cos((t*12-9-mese)*2*math.pi/12))*Tamp + Tmean + ampiezza_noise*math.cos(tau_noise*t*12)

    def T_return(t):
        return Tcos(t)/(Tamp+ampiezza_noise)*0.3*b_ref + 0.45*b_ref

    def Betaeq_return(t):
        return (Tmean-Tcos(t))/(Tamp+ampiezza_noise)*0.3*b_ref + 0.45*b_ref

    #[T,betaeq] = tg.generate_temp_by_adri(b_ref)
    [T,betaeq] = [T_return, Betaeq_return]
    dataset = dg.generate_dataset([[T,betaeq]], data)
    def fSIR(u,beta):
      S,I,R = u
      dS = -t_max*beta*S*I
      dI = t_max*(beta*S*I - a*I)
      dR = t_max*a*I
      return np.array([dS, dI, dR])

    from utilities import MyRK4
    beta0 = data['beta0']
    # for i in range(20):
    #     index = random.randint(1, len(dataset[0, :, 0])-1)
    #     t,sir = MyRK4.rk4(fSIR,[S0, I0, R0], 0, 1, N-1, dataset[1,index,])
    #     I = sir[:,1].copy()
    #     plt.plot(t,I)
    #     plt.show()

    t,sir = MyRK4.rk4(fSIR,[S0, I0, R0], 0, 1, N-1, dataset[1,0,])
    I = sir[:,1].copy()
    T_vec = np.zeros(shape = (N,))
    betaeq_vec = np.zeros(shape = (N,))

    for i in range(N):
        T_vec[i] = T(t[i])
        betaeq_vec[i] = betaeq(t[i])

    fig, (ax1, ax2, ax3) = plt.subplots(3)
    fig.suptitle('Temperatura, betaeq, infetti a partire dal mese ' + str(mese))
    ax1.plot(t, T_vec)
    ax1.legend("T(t)")
    ax2.plot(t, betaeq_vec)
    ax2.legend(["betaeq(t)"])
    ax3.plot(t, I)
    ax3.legend(["I(t)"])
    plt.show()