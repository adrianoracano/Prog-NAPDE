import numpy as np
import tensorflow as tf
import math
import MyCrankNicolsonClass as cnc
from matplotlib import pyplot as plt
import os
import random
import MyDatasetGenerator as dg

tfk = tf.keras
tfkl = tf.keras.layers
tf.keras.backend.set_floatx('float64')


seed = 42

random.seed(seed)
os.environ['PYTHONHASHSEED'] = str(seed)
np.random.seed(seed)
tf.random.set_seed(seed)
tf.compat.v1.set_random_seed(seed)

initializers = tf.keras.initializers
init = initializers.GlorotNormal(seed = seed)
tf.keras.layers.experimental.preprocessing.Normalization(dtype='float64')
model = tfk.Sequential([
    tfk.layers.Dense(5, activation='sigmoid', input_shape=(1,), kernel_initializer=init, bias_initializer=initializers.Zeros()),
    tfk.layers.Dense(5, activation='sigmoid', input_shape=(1,), kernel_initializer=init, bias_initializer=initializers.Zeros()),
  #    tfk.layers.Dense(20, activation='relu'),
    tfk.layers.Dense(1, activation='linear',kernel_initializer= init, bias_initializer=initializers.Zeros()),
#    tfk.layers.Softmax()
])
learning_rate = 0.01
optimizer = tf.optimizers.SGD(learning_rate)


N = 100
training_steps = 100
display_step = 1
def f1(x):
  return math.exp(-x)
summation = []
T = 10
dt = T/N
def loss1():
  y0 =  tf.constant([[f1(0)]], dtype = 'float64')
  for i in range(N-1):
    i = i + 1
    y = y0 + dt*i * model([[y0]])
    summation.append((f1(dt*i) - y))
    y0 = y
  return tf.reduce_mean(tf.abs(summation))

def train_step2():
  with tf.GradientTape() as tape:
    loss = loss1()
  gradients = tape.gradient(loss,model.trainable_variables)
  optimizer.apply_gradients(zip(gradients, model.trainable_variables))
  return gradients

#TRAINING LOOP
for i in range(training_steps):
  gradients = train_step2()
  if i % display_step == 0:
    print("iterazione numero: %i " %(i))
    print("loss: %f " % (loss1()))

gradients = train_step2()
gradients

model.trainable_variables

#PLOT
y = np.zeros(N)
y[0]= f1(0)
yex = y
curr_y = tf.constant([f1(0)], dtype = 'float64')
for i in range(N-1):
    # x = np.array([curr_beta, T(i*dt)])
    # print(curr_beta)
    next_y = curr_y + dt * model(curr_y)
    y[i+1] = next_y.numpy()[0][0]
    yex[i+1] = f1(dt*i)
    curr_y = next_y

t = np.linspace(start = 0, stop = 10, num = N)
plt.plot(t, y)
plt.plot(t, yex)
