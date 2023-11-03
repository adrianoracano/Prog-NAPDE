import tensorflow as tf
from tensorflow import keras as tfk
from keras import layers as tfkl
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

from Data import *
from arguments import *
from utilities import MyRK4

def f(self, beta, Betaeq):
    return(1/tau)*(Betaeq - beta)*t_max
def fSIR(self, u, beta):
    S, I, R = u
    dS = -t_max * beta * S * I / TOT
    dI = t_max * (beta * S * I / TOT - alpha * I)
    dR = t_max * alpha * I
    return np.array([dS, dI, dR])

def generate_I(self, beta_vec):
    I = np.zeros([K, N])
    for k in range(K):
        t, sir = MyRK4.rk4(self.fSIR, [S0, I0, R0], 0, 1, N - 1, beta_vec)
        I[k,] = sir[:, 1].copy()
        return t,I


