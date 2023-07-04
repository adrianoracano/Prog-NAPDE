# -*- coding: utf-8 -*-
"""
Created on Tue Jul  4 11:30:45 2023

@author: alean
"""
import tensorflow as tf

def forward_euler_step(curr_S_nn, curr_I_nn, curr_beta, dt, a):
    next_S_nn = curr_S_nn - dt*tf.matmul(curr_beta, tf.matmul(curr_S_nn, curr_I_nn))
    next_I_nn = curr_I_nn + dt*(tf.matmul(curr_beta, tf.matmul(curr_S_nn, curr_I_nn))) - dt*a*curr_I_nn
    return next_S_nn, next_I_nn
"""
next_S_nn = curr_S_nn - dt*tf.matmul(curr_y, tf.matmul(curr_S_nn, curr_I_nn))
next_I_nn = curr_I_nn + dt*(tf.matmul(curr_y, tf.matmul(curr_S_nn, curr_I_nn))) - dt*a*curr_I_nn
"""

def runge_kutta_step(curr_s, curr_i, curr_beta, next_beta, dt, a):
    t=0.
    s = curr_s
    i = curr_i
    m1 = -tf.matmul(curr_beta, tf.matmul(s, i))
    k1 = tf.matmul(curr_beta, tf.matmul(s, i)) - a*i
    l1 = a*i
    #def m2,k2,l2
    ft2 = t + (dt/2.)
    fs2 =  s + (dt/2.)*m1
    fi2 =  i + (dt/2.)*k1
    b = 0.5*( tf.add(curr_beta, next_beta) )
    m2 = -tf.matmul(b, tf.matmul(fs2, fi2))
    k2 = tf.matmul(b, tf.matmul(fs2, fi2)) - a*fi2
    l2 = a*fi2
    #def m3,k3,l3
    ft3 = t + (dt/2.)
    fs3 =  tf.add(s, (dt/2.)*m2)
    fi3 =  tf.add(i, (dt/2.)*k2)
    b = 0.5*(tf.add(curr_beta, next_beta))
    m3 = -tf.matmul(b, tf.matmul(fs3, fi3)) # -b*fs3*fi3
    k3 =  tf.matmul(b, tf.matmul(fs3, fi3)) - a*fi3 # b*fs3*fi3 - a*fi3
    l3 = a*fi3
    #def m4,k4,l4
    ft4 = t + dt
    fs4 =  tf.add(s, dt*m3)
    fi4 =  tf.add(i, dt*k3)
    b = next_beta
    m4 = -tf.matmul(b, tf.matmul(fs4, fi4))# -b*fs4*fi4
    k4 =  tf.matmul(b, tf.matmul(fs4, fi4)) - a*fi4 # b*fs4*fi4 - a*fi4
    l4 = a*fi4
    t = t + dt
    next_s = s + (dt/6.)*(m1 + 2.*m2 +2.*m3 + m4)
    next_i = i + (dt/6.)*(k1 + 2.*k2 +2.*k3 + k4)
    return next_s, next_i