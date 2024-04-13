# -*- coding: utf-8 -*-
"""
Created on Mon Apr 10 12:29:31 2023

@author: alean
"""

def MyCrankNicolsol(sys, x0, T, N, max_it=15, toll=1e-6, h=1e-2, step=1e-2):
    import numpy as np
    import MyNewton as mn
    t=np.linspace(0, T, num=N)
    dt=T/N
    n=len(x0)
    u=np.zeros((n, N))
    u[:,0]=x0.copy()
    curr_it=0
    while curr_it<N-1:
        curr_t=t[curr_it]
        curr_u = u[:,curr_it]
        def sys_for_newton(x):
            return x-curr_u-dt*0.5*( mn.compute_sys(sys, x, curr_t+dt)\
                                    +mn.compute_sys(sys, curr_u, curr_t) )
        curr_it=curr_it+1
        u[:,curr_it]=mn.my_newton(sys_for_newton, curr_u, max_it, toll, h, step)
    del np
    return (t, u)
