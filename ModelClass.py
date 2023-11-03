import tensorflow as tf
from tensorflow import keras as tfk
from keras import layers as tfkl
import os
import numpy as np
import MyTemperatureGenerator as tg
import MyDatasetGenerator as dsg
from utilities import HerRungeKutta as hrk
import argparse
import sys
import pickle
from matplotlib import pyplot as plt
from utilities import SirHelperFunctions as shf
import MyDatasetGenerator as dg
import random
import math
tf.keras.backend.set_floatx('float64')

from Data import *
from arguments import *

class Model:
    def __init__(self, n_hidden = n_hidden, addDropout = False ,addBNorm = True, load_path = '', save_path = ''):
        self.n_input = 2
        self.n_hidden = n_hidden
        if(len(load_path)>0):
            print("Loading model from " + args.load_model + "...\n")
            self.load_model(load_path)
        else:
            print("Generating new model with standard parameters ...\n")
            model = tfk.Sequential()
            model.add(tfkl.Dense(
                units=n_hidden,
                input_dim=self.n_input,
                activation=None,
                use_bias=True,
                kernel_initializer="glorot_uniform",
                bias_initializer="zeros",
                kernel_regularizer=None,
                bias_regularizer=None,
                activity_regularizer=None,
                kernel_constraint=None,
                bias_constraint=None))
            if (addBNorm):
                model.add(tfkl.BatchNormalization())
            model.add(tfkl.Activation('sigmoid'))
            if (addDropout):
                model.add(tfkl.Dropout(0.3))
            model.add(tfkl.Dense(1))
            if (addBNorm):
                model.add(tfkl.BatchNormalization())
            self.model = model
        self.optimizer = tf.optimizer.Adam(learning_rate)
        self.n_iter = 0

    def g(self, y, v):
        y = (1.0/b_ref)*y
        v = (1.0/b_ref)*v
        v.shape = (v.shape[0], 1)
        tv = tf.constant(v, dtype='float64')
        x = tf.concat([y, tv], 1)
        return self.model(x)
    def custom_loss(self, K, dataset, I, sir_0):
        summation = []
        curr_beta = tf.constant(0.05 * np.ones([K, 1], dtype='float64'), dtype='float64')
        curr_I_nn = tf.constant(sir_0[1] * np.ones([K, 1], dtype='float64'), dtype='float64')
        curr_S_nn = tf.constant(sir_0[0] * np.ones([K, 1], dtype='float64'), dtype='float64')
        for i in range(N - 1):
            next_beta = curr_beta + dt * self.g(curr_beta, dataset[0, :, i])
            if solver == 'ea':
                next_S_nn = curr_S_nn - t_max * dt * curr_beta * curr_S_nn * curr_I_nn / TOT
                next_I_nn = curr_I_nn + t_max * dt * curr_beta * curr_S_nn * curr_I_nn / TOT - t_max * dt * alpha * curr_I_nn
            if solver == 'rk':
                next_S_nn, next_I_nn = shf.runge_kutta_step(curr_S_nn, \
                                                            curr_I_nn, curr_beta, next_beta, dt, alpha)
            if i % step_summation == 0:
                I_exact = I[:, i + 1]
                I_exact.shape = (I_exact.shape[0], 1)
                summation.append(tf.reduce_sum((next_I_nn - I_exact) ** 2))
            curr_beta = next_beta
            curr_S_nn = next_S_nn
            curr_I_nn = next_I_nn
        return tf.reduce_sum(summation)

    def train_step(self, K ,dataset, I):
        with tf.GradientTape() as tape:
            loss = self.custom_loss(K, dataset, I)
        trainable_variables = self.model.trainable_variables
        gradients = tape.gradient(loss, trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, trainable_variables))

    def train(self, dataset, I, val_set, I_val, max_iter = training_steps):
        print("Starting the training...\n")
        loss_history = np.zeros(int(training_steps / display_step))
        i_history = 0
        #if args.validate:
        loss_history_val = np.zeros(int(training_steps / display_step))

        try:
            for i in range(max_iter):
                self.train_step(K, dataset, I)
                if i % display_step == 0:
                    print("iterazione %i:" % i)
                    loss_history[i_history] = self.custom_loss(K, dataset, I)
                    print("loss on training set: %f " % loss_history[i_history])
                    if args.validate:
                        loss_history_val[i_history] = self.custom_loss(K_val, val_set, I_val)
                        print("loss on validation set: %f" % loss_history_val[i_history])
                    i_history = i_history + 1
                if i % display_weights == 0:
                    print("pesi all'iterazione %i:")
                    print(self.model.trainable_variables)
                self.n_iter = i
        except KeyboardInterrupt:
            print('\nTraining interrupted by user. Proceeding to save the weights and plot the solutions...\n')

    def load_model(self, load_path):
        print("Loading model from " + load_path + "...\n")
        self.model = tfk.models.load_model(load_path)

    def save_model(self, save_path):
        print("Saving the model in " + save_path + "...\n")
        self.model(save_path + "iter" + str(self.n_iter))
    # if len(args.load_model) > 0 and args.new_model:
    #     print("Cannot generate new model and loading an existing one. Aborting..\n")
    #     sys.exit()
    # if (args.new_model):
    #     model = get_model()
    # elif(args.load_model):
    #     print("Loading model from " + args.load_model + "...\n")
    #     model = tfk.models.load_model(args.load_model)
    # def g(y, v):
    #     # y = (1.0/b_ref)*y
    #     # v = (1.0/b_ref)*v
    #     v.shape = (v.shape[0], 1)
    #     tv = tf.constant(v, dtype='float64')
    #     x = tf.concat([y, tv], 1)
    #     return model(x)