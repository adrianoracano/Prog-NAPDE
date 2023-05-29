# -*- coding: utf-8 -*-
"""
Created on Sun May 21 16:40:50 2023

@author: alean
"""
import random
import math

def generate_temperature(fun_type, t_max=1.0, T_max=1.0):
    if fun_type == "step":
        t1 = random.uniform(0., t_max)
        def T(t):
            d = min(t1, t_max-t1)
            d=d/5
            if t<t1:
                return T_max
            else:
                return math.exp(-( (t-t1)/d )**2)
        return T
    if fun_type == "exp":
        mean = random.uniform(0, t_max)
        width = t_max/2 + random.uniform(-t_max/4, t_max/4)
        def T(t):
            return math.exp(-( (t-mean)/width )**2)
        return T
    if fun_type == "decr-exp":
        tau = random.gauss(0.5*t_max, 0.3*t_max)
        def T(t):
            return math.exp( -t/tau )
        return T

    