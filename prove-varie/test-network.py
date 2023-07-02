# -*- coding: utf-8 -*-
"""
Created on Sun May 14 21:39:19 2023

@author: alean
"""

center = random.gauss(0.5,0.1)
height = random.gauss(16.0, 2.0)
width = random.gauss(14.0, 1.0)
def T(t):
    return math.sin(2*math.pi/t_max*(t-center))*width + height



def dbeta(beta, t):
    return f(beta, T(t))
sys = [dbeta]
cn_solver = cnc.CrankNicolson(sys, beta0, t_max, N)
cn_solver.compute_solution()
t, beta = cn_solver.get_solution()
curr_beta = tf.constant([[beta0[0]]])
beta_hat = np.zeros(beta.shape[1])
beta_hat[0]=curr_beta.numpy()[0][0]
for i in range(beta.shape[1]-1):
    next_beta = curr_beta + dt * g( curr_beta, T(dt*i) )
    curr_beta = next_beta
    beta_hat[i+1] = next_beta.numpy()[0][0]
pp = plt.plot(t, beta_hat)
plt.plot(t, beta[0, ])
plt.legend(["soluzione rete", "solzione vera"])