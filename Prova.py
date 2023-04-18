# -*- coding: utf-8 -*-
"""
Created on Sun Mar 19 10:59:28 2023

@author: alean
"""

# import matplotlib.pyplot as plt
import numpy as np
import matplotlib.pyplot as plt
import math

def beta(t):
    return math.sin(t)
def alpha(t):
    return 3.0
def dS(sir, t):
    return -beta(t)*sir[0]*sir[1]
def dI(sir, t):
    return beta(t)*sir[0]*sir[1]-alpha(t)*sir[1]
def dR(sir, t):
    return alpha(t)*sir[1]
sir_0=np.array([400., 10., 0.])

sys=(dS, dI, dR)
T=10.
N=500
legend=["S", "I", "R"]

def dx(xyz, t):
    return 10.0*(xyz[1]-xyz[0])
def dy(xyz, t):
    return 28.0*xyz[0]-xyz[0]*xyz[2]-xyz[1]
def dz(xyz, t):
    return xyz[0]*xyz[1]-2.7*xyz[2]

sys_lorenz=(dx, dy, dz)
xyz_0=np.array([1., 1., 1.])


import MyCrankNicolsonClass as cnc

cnsolver= cnc.CrankNicolson(sys_lorenz, xyz_0, T, N)
cnsolver.compute_solution()
cnsolver.plot_solutions(legend)
cnsolver.plot_3d()


del cnc, np, plt, math