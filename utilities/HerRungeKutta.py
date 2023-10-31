# -*- coding: utf-8 -*-

"""

Ora Runge Kutta è una funzione: per usarla basta scrivere in uno script qualsiasi:
    
import HerRungeKutta as rk
import math
def beta(t):
    return math.sin(t)  # o una qualsiasi funzione a scelta
a=3.
T=5.
sir_0=(400., 10., 0.)
h=0.05

rk.RungeKutta(sir_0, beta, h, T, a)

e la funzione calcola la soluzione e la plotta
Volendo si può modificare la funzione in modo da farle restituire l'array con la
soluzione trovata o cose simili.


"""

def RungeKutta(sir_0, beta, N, T, a):
    import numpy as np
    dt = 1.0/N
    s_old, i_old, r_old = sir_0
    i = np.zeros(N)
    s = np.zeros(N)
    r = np.zeros(N)
    i[0] = i_old
    s[0] = s_old
    r[0] = r_old
    for k in range(N-1):
        s[k+1] = s_old - dt*T*s_old*i_old*beta[k]
        i[k+1] = i_old +dt*T*s_old*i_old*beta[k] - T*a*i_old*dt
        i_old = i[k+1]
        s_old = s[k+1]
        r[k+1] = 1-i[k+1]-s[k+1]
    return (s, i, r)
    import matplotlib.pyplot as plt 
    import numpy as np
    h = 1.0/N      # T periodo, N numero di timesteps
    S = np.zeros(N) # Si va da t_0 a t_(N-1)
    I = np.zeros(N)
    R = np.zeros(N)
    S[0] = sir_0[0]
    I[0] = sir_0[1]
    R[0] = sir_0[2]
    t = 0.
    k = 0
    while(k<N-1):
        #def m1,k1,l1
        s = S[k] # al tempo k, per k = 0,...,N-1
        i = I[k]
        r = R[k]
        m1 = -T*beta[k]*s*i
        k1 = T*beta[k]*s*i - T*a*i
        l1 = T*a*i
        #def m2,k2,l2
        ft2 = t + (h/2.) #t_(k + 1/2)
        fs2 =  s + (h/2.)*m1
        fi2 =  i + (h/2.)*k1
        fr2 =  r + (h/2.)*l1
        b = 0.5*(beta[k+1] + beta[k]) # beta(t_(k + 1/2))
        m2 = -T*b*fs2*fi2
        k2 = T*b*fs2*fi2 - T*a*fi2
        l2 = T*a*fi2
        #def m3,k3,l3
        ft3 = t + (h/2.)
        fs3 =  s + (h/2.)*m2
        fi3 =  i + (h/2.)*k2
        fr3 =  r + (h/2.)*l2
        b = 0.5*(beta[k+1]+beta[k]) 
        m3 = -T*b*fs3*fi3
        k3 = T*b*fs3*fi3 - T*a*fi3
        l3 = T*a*fi3
        #def m4,k4,l4
        ft4 = t + h
        fs4 =  s + h*m3
        fi4 =  i + h*k3
        fr4 =  r + h*l3
        b = beta[k+1]
        m4 = -T*b*fs4*fi4
        k4 = T*b*fs4*fi4 - T*a*fi4
        l4 = T*a*fi4
        t = t + h
        S[k+1] = s + (h/6.)*(m1 + 2.*m2 +2.*m3 + m4)
        I[k+1] = i + (h/6.)*(k1 + 2.*k2 +2.*k3 + k4)
        R[k+1] = r + (h/6.)*(l1 + 2.*l2 +2.*l3 + l4)
        k = k+1
    return (S, I, R)