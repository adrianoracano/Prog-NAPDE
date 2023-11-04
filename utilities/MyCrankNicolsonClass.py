# -*- coding: utf-8 -*-
"""
Created on Sat Apr 15 18:45:15 2023

@author: alean
"""

import numpy as np

class CrankNicolson:
    def __init__(self, sys,x0,T,N,max_it=15,toll=1e-6,h=1e-2,step=1e-2):
        self.sys = sys
        self.x0 = x0
        self.max_it=max_it
        self.toll=toll
        self.h=h
        self.step=step
        self.T=T
        self.N=N
        self.t=np.linspace(0, T, num=N)
        self.u=np.zeros((len(x0), N))
        
        
    def compute_solution(self):
        import MyNewton as mn
        dt=self.T/self.N
        self.u[:, 0]=self.x0.copy()
        curr_it=0
        while curr_it<self.N-1:
            curr_t=self.t[curr_it]
            curr_u=self.u[:,curr_it]
            def sys_for_newton(x):
                return x-curr_u-dt*0.5*( mn.compute_sys(self.sys, x, curr_t+dt)\
                                        +mn.compute_sys(self.sys, curr_u, curr_t) )
            curr_it=curr_it+1
            self.u[:,curr_it]=mn.my_newton(sys_for_newton, curr_u, \
                                      self.max_it, self.toll, self.h, self.step)
        del mn
    
    def get_solution(self):
        return (self.t, self.u)
    
    def plot_solutions(self, legend=[]):
        import matplotlib.pyplot as plt
        for i in range(len(self.x0)):
            plt.plot(self.t, self.u[i,:])
        if len(legend)==len(self.x0):
            plt.legend(legend)
        plt.show()
        del plt
    def plot_3d(self, legend=[]):
        if len(self.x0)==3:
            import matplotlib.pyplot as plt
            ax = plt.figure().add_subplot(projection='3d')
            ax.plot(self.u[0,:], self.u[1,:], self.u[2,:])
            if len(legend)==3:
                plt.legend(legend)
            plt.show()
            del plt


        
        

            
        
        
        