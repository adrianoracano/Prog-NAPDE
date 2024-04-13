# -*- coding: utf-8 -*-
"""
Created on Tue Jul  4 16:25:45 2023

@author: alean
"""
def g(t):
    return 1/(1+math.exp(-t))
def dg(t):
    return math.exp(-t)/((1+math.exp(-t))**2)

weights = {
'h1': tf.Variable([[1.0]])
}


def multilayer_perceptron(x):
  layer_1 = tf.matmul(x, weights['h1'])
  output = tf.nn.sigmoid(layer_1)
  return output

def funzione(X):
    X = X + 2.0*multilayer_perceptron(X)
    X = X +2.0*multilayer_perceptron(X)
    summation = []
    summation.append(X**2)
    summation.append(X**2)
    total_summation = []
    total_summation.append(tf.reduce_sum(summation))
    total_summation.append(tf.reduce_sum(summation))
    return tf.reduce_sum(total_summation)

def f(x):
    X = tf.constant([[x]])
    X = funzione(X)
    return X
    

with tf.GradientTape() as tape:
    df = f(1.0)
trainable_variables=weights.values()
gradients = tape.gradient(df, trainable_variables)
print(gradients)
