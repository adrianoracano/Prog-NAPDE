import numpy as np
from utilities import MyRK4

"""
def f(beta, Betaeq):
    return(1/tau)*(Betaeq - beta)*t_max
"""

def generate_I(beta_vec, t_max, alpha, sir_0):
    K = beta_vec.shape[0]
    N = beta_vec.shape[1]
    I = np.zeros([K, N])
    
    def fSIR(u, beta):
        S, I, R = u
        dS = -t_max * beta * S * I
        dI = t_max * (beta * S * I - alpha * I)
        dR = t_max * alpha * I
        return np.array([dS, dI, dR])
    S0, I0, R0 = sir_0
    for k in range(K):
        t, sir = MyRK4.rk4(fSIR, [S0, I0, R0], 0, 1, N - 1, beta_vec)
        I[k,] = sir[:, 1].copy()
        return t,I


