# -*- coding: utf-8 -*-
"""
Created on Sun May 21 17:14:10 2023

@author: alean
"""

T = tg.generate_temperature("exp")
y_real = np.zeros(N)
y_nn = np.zeros(N)
y_real[0] = y0
y_nn[0] = y0 
curr_y = tf.constant([[y0]], dtype = 'float32')
for i in range(N-1):
    next_y = curr_y + dt*g(curr_y, T(t[i]))
    y_nn[i+1] = next_y.numpy()[0][0]
    y_real[i+1] = y_real[i] + dt*f(y_real[i], T(t[i]))
    curr_y = next_y

plt.plot(t, y_real)
plt.plot(t, y_nn)
plt.legend(["soluzione reale", "soluzione rete"])