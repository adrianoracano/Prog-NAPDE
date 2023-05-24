# -*- coding: utf-8 -*-
"""
Created on Thu May 18 22:36:22 2023

@author: alean
"""

# -*- coding: utf-8 -*-
"""
#############
QUESTA PROVA NON USA IL TEMPO
############
"""
import tensorflow as tf
import numpy as np
from matplotlib import pyplot as plt
import math
import MyCrankNicolsonClass as cnc

learning_rate = 0.01
n_input = 1
n_hidden = 5
n_output = 1
display_step = 10
weights = {
'h1': tf.Variable(tf.random.normal([n_input, n_hidden])),
'out': tf.Variable(tf.random.normal([n_hidden, n_output]))
}
biases = {
'b1': tf.Variable(tf.random.normal([n_hidden])),
'out': tf.Variable(tf.random.normal([n_output]))
}
# Stochastic gradient descent optimizer.
optimizer = tf.optimizers.SGD(learning_rate)

# Create model
def multilayer_perceptron(x):
  layer_1 = tf.add(tf.matmul(x, weights['h1']), biases['b1'])
  layer_1 = tf.nn.sigmoid(layer_1)
  output = tf.matmul(layer_1, weights['out']) + biases['out']
  return output

def f(x):
    return math.exp(-3.0*x)
x = np.linspace(0, 1, 50)
x.shape=(50, 1)
y_real = np.zeros([50, 1])
for i in range(x.shape[0]):
    y_real[i, 0]=f(x[i, 0])
    
dt = 1/50

def dy(y, t):
    return 10.*(y**2-y)
"""
sys = [dy]
y0 = np.array([0.5])
cn_solver = cnc.CrankNicolson(sys, y0, 1.0, 50)
cn_solver.compute_solution()
t, y_real = cn_solver.get_solution()
"""
y_real=np.zeros(50)
y_real[0] = 0.5
for i in range(len(y_real)-1):
    y_real[i+1] = y_real[i] + dt*dy(y_real[i], 0.)

    



def custom_loss():
    tx = tf.constant(x, dtype = 'float32')
    summation = []
    y0 = tf.constant([[1.0]], dtype = 'float32')
    curr_y = y0
    for i in range(x.shape[0] - 1):
        next_y = curr_y + dt*multilayer_perceptron(curr_y)
        # real_value = tf.con
        summation.append((next_y - y_real[i+1])**2)
        curr_y=next_y
    return tf.reduce_sum(summation)
    

def train_step():
    with tf.GradientTape() as tape:
        loss = custom_loss()
    trainable_variables=list(weights.values())+list(biases.values())
    gradients = tape.gradient(loss, trainable_variables)
    optimizer.apply_gradients(zip(gradients, trainable_variables))
    
training_steps = 1000
display_step = 50

for i in range(training_steps):
  train_step()
  if i % display_step == 0:
    print("iterazione %i:" % i)
    print("loss: %f " % (custom_loss()))
    
y_nn = np.zeros(50)
y0 = tf.constant([[f(0.)]], dtype = 'float32')
curr_y = y0
y_nn[0] = f(0.)
for i in range(x.shape[0] - 1):
    next_y = curr_y + dt*multilayer_perceptron(curr_y)
    # real_value = tf.con
    curr_y=next_y
    y_nn[i+1] = next_y.numpy()[0][0]
    
t=np.linspace(0., 1., 50)
plt.plot(t, y_nn)
plt.plot(t, y_real)
plt.legend(["soluzione rete", "soluzione vera"])
    
    



