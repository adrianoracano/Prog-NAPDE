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

def RungeKutta(sir_0, beta, h, T, a):
    import matplotlib.pyplot as plt 
    s = sir_0[0]
    i = sir_0[1]
    r = sir_0[2]
    t = 0.
    b = beta(t)
    print('t','\t','\t','\t','s','\t','\t','i','\t','\t','r','\n')
    print(round(t,2),'\t','\t','\t',round(s,2),'\t','\t',i,'\t','\t',round(r,2))
    while(t<T):
        #def m1,k1,l1
        m1 = -b*s*i
        k1 = b*s*i - a*i
        l1 = a*i
        #def m2,k2,l2
        ft2 = t + (h/2)
        fs2 =  s + (h/2.)*m1
        fi2 =  i + (h/2.)*k1
        fr2 =  r + (h/2.)*l1
        b = beta(ft2)
        m2 = -b*fs2*fi2
        k2 = b*fs2*fi2 - a*fi2
        l2 = a*fi2
        #def m3,k3,l3
        ft3 = t + (h/2.)
        fs3 =  s + (h/2.)*m2
        fi3 =  i + (h/2.)*k2
        fr3 =  r + (h/2.)*l2
        b = beta(ft3)
        m3 = -b*fs3*fi3
        k3 = b*fs3*fi3 - a*fi3
        l3 = a*fi3
        #def m4,k4,l4
        ft4 = t + h
        fs4 =  s + h*m3
        fi4 =  i + h*k3
        fr4 =  r + h*l3
        b = beta(ft4)
        m4 = -b*fs4*fi4
        k4 = b*fs4*fi4 - a*fi4
        l4 = a*fi4
        t = t + h
        s = s + (h/6.)*(m1 + 2.*m2 +2.*m3 + m4)
        i = i + (h/6.)*(k1 + 2.*k2 +2.*k3 + k4)
        r = r + (h/6.)*(l1 + 2.*l2 +2.*l3 + l4)
        print(round(t,2),'\t','\t','\t',round(s,2),'\t','\t',round(i,2),'\t','\t',round(r,2))
        #print(s+r+i)
        plt.plot(t, s, 'bo') 
        plt.plot(t, i, 'ro') 
        plt.plot(t, r, 'go') 
    plt.show()