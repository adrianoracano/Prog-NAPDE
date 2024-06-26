import tensorflow as tf
from tensorflow import keras as tfk
from keras import layers as tfkl
import numpy as np
import pickle
import os

tf.keras.backend.set_floatx('float64')


class Model:
    def __init__(self, n_input, n_hidden, learning_rate, b_ref, addDropout = False ,\
                 addBNorm = True, load_path = '', use_keras = False, numLayers = 1):
        self.n_input = n_input
        self.n_hidden = n_hidden
        self.numLayers = numLayers
        self.b_ref = b_ref
        self.n_output = 1
        self.use_keras = use_keras
        self.n_iter = 0
        if(len(load_path)>0):
            print("Loading model from " + load_path + "...\n")
            self.load_model(load_path)
            self.n_hidden = len(self.weights['h1'][0])
            self.numLayers = len(self.weights) - 1
            print('len(weights) of the loaded model is ' + str(numLayers))
            print('number of layers of the loaded model is ' + str(self.numLayers))
        else:
            if(self.numLayers == 1):
              self.weights = {
              'h1': tf.Variable(tf.random.normal([self.n_input, self.n_hidden], dtype='float64'), dtype='float64'),
              #'h2': tf.Variable(tf.random.normal([self.n_input, n_hidden], dtype='float64'), dtype='float64'),
              'out': tf.Variable(tf.random.normal([self.n_hidden, self.n_output], dtype='float64'), dtype='float64')
              }
              self.biases = {
              'b1': tf.Variable(tf.random.normal([self.n_hidden], dtype='float64'), dtype='float64'),
              #'b2': tf.Variable(tf.random.normal([n_hidden], dtype='float64'), dtype='float64'),
              'out': tf.Variable(tf.random.normal([self.n_output], dtype='float64'), dtype='float64')
              }
              print("Generating new model with standard parameters ...\n")
            elif(self.numLayers == 2):
              self.weights = {
              'h1': tf.Variable(tf.random.normal([self.n_input, self.n_hidden], dtype='float64'), dtype='float64'),
              'h2': tf.Variable(tf.random.normal([self.n_hidden, self.n_hidden], dtype='float64'), dtype='float64'),
              'out': tf.Variable(tf.random.normal([self.n_hidden, self.n_output], dtype='float64'), dtype='float64')
              }
              self.biases = {
              'b1': tf.Variable(tf.random.normal([self.n_hidden], dtype='float64'), dtype='float64'),
              'b2': tf.Variable(tf.random.normal([self.n_hidden], dtype='float64'), dtype='float64'),
              'out': tf.Variable(tf.random.normal([self.n_output], dtype='float64'), dtype='float64')
              }
              print("Generating new model with standard parameters ...\n")
        if self.numLayers  == 1:
            self.multilayer_perceptron = self.multilayer_perceptron_1layer
        elif self.numLayers == 2:
            self.multilayer_perceptron = self.multilayer_perceptron_2layers
            
        self.optimizer = tf.optimizers.Adam(learning_rate)
        self.keras_model = tfk.Sequential()
        self.keras_model.add(tfkl.Dense(
            units=self.n_hidden,
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
        self.keras_model.add(tfkl.BatchNormalization())
        self.keras_model.add(tfkl.Dense(
            units=self.n_hidden,
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
        self.keras_model.add(tfkl.BatchNormalization())
        self.keras_model.add(tfkl.Activation('sigmoid'))
        if addDropout:
            self.keras_model.add(tfkl.Dropout(0.3))
        self.keras_model.add(tfkl.Dense(1))
        self.keras_model.add(tfkl.BatchNormalization())
        
        self.loss = self.custom_loss

    def multilayer_perceptron_1layer(self, x):
        layer_1 = tf.add(tf.matmul(x, self.weights['h1']), self.biases['b1'])
        layer_1 = tf.nn.sigmoid(layer_1)
        #layer_2 = tf.add(tf.matmul(x, self.weights['h2']), self.biases['b2'])
        #layer_2 = tf.nn.sigmoid(layer_2)
        output = tf.matmul(layer_1, self.weights['out']) + self.biases['out']
        return output
    def multilayer_perceptron_2layers(self, x):
        layer_1 = tf.add(tf.matmul(x, self.weights['h1']), self.biases['b1'])
        layer_1 = tf.nn.sigmoid(layer_1)
        layer_2 = tf.add(tf.matmul(layer_1, self.weights['h2']), self.biases['b2'])
        layer_2 = tf.nn.sigmoid(layer_2)
        output = tf.matmul(layer_2, self.weights['out']) + self.biases['out']
        return output
    def g(self, y, v):
        v.shape = (v.shape[0], self.n_input-1)
        tv = tf.constant(v, dtype='float64')
        x = tf.concat([y, tv], 1)
        if self.use_keras:
            return self.keras_model(x)
        else:
            return self.multilayer_perceptron(x)
    def custom_loss(self, dataset, I, t_max, alpha, beta0):
        summation = []
        K = I.shape[0]
        N = I.shape[1]
        b = np.zeros([K, 1], dtype = 'float64')
        b[:, 0] = beta0 # potrebbe dare problemi?
        curr_beta = tf.constant(b, dtype='float64')
        # curr_I_nn = tf.constant(sir_0[1] * np.ones([K, 1], dtype='float64'), dtype='float64')
        curr_I_nn = tf.constant( I[:, 0], dtype='float64', shape = (K, 1))
        # curr_S_nn = tf.constant(sir_0[0] * np.ones([K, 1], dtype='float64'), dtype='float64')
        curr_S_nn = tf.constant(1.0-I[:, 0], dtype='float64', shape = (K, 1))
        dt = 1.0/N
        for i in range(N - 1):
            next_beta = curr_beta + t_max *dt * self.g(curr_beta, dataset[:, i])
            # step di eulero in avanti
            next_S_nn = curr_S_nn - t_max * dt *  curr_beta * curr_S_nn * curr_I_nn
            next_I_nn = curr_I_nn + t_max * dt * ( curr_beta * curr_S_nn * curr_I_nn -  alpha * curr_I_nn )
            I_exact = I[:, i + 1]
            dI_exact = (I[:, i + 1] - I[:, i])/dt
            dI_nn = t_max * ( curr_beta * curr_S_nn * curr_I_nn - alpha * curr_I_nn )
            I_exact.shape = (I_exact.shape[0], 1)
            dI_exact.shape = (dI_exact.shape[0], 1)
            summation.append(tf.reduce_mean(tf.abs(next_I_nn - I_exact)))
            curr_beta = next_beta
            curr_S_nn = next_S_nn
            curr_I_nn = next_I_nn
        return tf.reduce_sum(summation)

    def train_step(self, dataset, I, t_max, alpha, beta0):
        with tf.GradientTape() as tape:
            loss = self.loss(dataset, I, t_max, alpha, beta0)
        # trainable_variables = self.model.trainable_variables
        if self.use_keras:
            trainable_variables = self.keras_model.trainable_variables
        else:
            trainable_variables = list(self.weights.values())+list(self.biases.values())
        gradients = tape.gradient(loss, trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, trainable_variables))
        self.n_iter = self.n_iter + 1

    def load_model(self, load_path):
        print("Loading model from " + load_path + "...\n")
        # self.model = tfk.models.load_model(load_path)
        if self.use_keras:
            self.keras_model = tfk.models.load_model(load_path)
        else:
            with open(load_path, 'rb') as file:
                self.weights, self.biases, self.n_iter = pickle.load(file)
    def save_model(self, save_path):
        print("Saving the model in " + save_path + "...\n")
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        save_path += '/model.pkl'
        if self.use_keras:
            self.keras_model.save(save_path + "iter" + str(self.n_iter))
        else:
            with open(save_path, 'wb') as file:
                pickle.dump((self.weights, self.biases, self.n_iter), file)

class NetworkForSIR:
    def __init__(self, model, display_step, t_max, alpha):
        self.model = model
        self.display_step = display_step
        self.t_max = t_max
        self.alpha = alpha

    def train(self, dataset, I,val_set, I_val, beta0_train, beta0_val, max_iter, display_weights, validate = True):
        display_step = self.display_step
        print("Starting the training...\n")
        loss_history = np.zeros(int(max_iter / display_step))
        i_history = 0
        #if args.validate:
        loss_history_val = np.zeros(int(max_iter / display_step))
        try:
            for i in range(max_iter):
                self.model.train_step(dataset, I, self.t_max, self.alpha, beta0_train)
                if i % display_step == 0:
                    print("iterazione %i:" % i)
                    loss_history[i_history] = self.model.loss(dataset, I,\
                                                                     self.t_max, self.alpha, beta0_train)
                    print("loss on training set: %f " % loss_history[i_history])
                    if validate:
                        loss_history_val[i_history] = self.model.loss(val_set, I_val,\
                                                                         self.t_max, self.alpha, beta0_val)
                        print("loss on validation set: %f" % loss_history_val[i_history])
                    i_history = i_history + 1

        except KeyboardInterrupt:
            print('\nTraining interrupted by user. Proceeding to save the weights and plot the solutions...\n')
        return loss_history, loss_history_val, i_history
    # volendo si possono aggiungere qui metodi per fare plot e cose simili

    def compute_beta_I(self, dataset, I0, beta0):
        N = dataset.shape[1]
        K = dataset.shape[0]
        b = np.zeros([K, 1], dtype = 'float64')
        b[:, 0] = beta0 # potrebbe dare problemi?
        curr_beta = tf.constant(b, dtype='float64')
        curr_I_nn = tf.constant( I0, dtype='float64', shape = (K, 1))
        curr_S_nn = tf.constant(1.0-I0, dtype='float64', shape = (K, 1))
        dt = 1.0/N
        beta_return = np.zeros([K, N], dtype='float64')
        I_return = np.zeros([K, N], dtype='float64')
        beta_return[:, 0] = beta0
        I_return[:, 0] = I0
        for i in range(N-1):
            next_beta = curr_beta + self.t_max * dt * self.model.g(curr_beta, dataset[:, i])
            beta_return[:, i+1] = next_beta.numpy()[:, 0].copy()
            # step di eulero in avanti
            next_S_nn = curr_S_nn - self.t_max * dt * curr_beta * curr_S_nn * curr_I_nn
            next_I_nn = curr_I_nn + self.t_max * dt * ( curr_beta * curr_S_nn * curr_I_nn -  self.alpha * curr_I_nn )
            I_return[:, i+1] = next_I_nn.numpy()[:, 0].copy()
            # update
            curr_S_nn = next_S_nn
            curr_beta = next_beta
            curr_I_nn = next_I_nn
        return beta_return, I_return

    def compute_beta_I_R(self, dataset, I0, R0, beta0):
        N = dataset.shape[1]
        K = dataset.shape[0]
        b = np.zeros([K, 1], dtype = 'float64')
        b[:, 0] = beta0 # potrebbe dare problemi?
        curr_beta = tf.constant(b, dtype='float64')
        curr_I_nn = tf.constant( I0, dtype='float64', shape = (K, 1))
        curr_R_nn = tf.constant( R0, dtype='float64', shape = (K, 1))
        curr_S_nn = tf.constant(1.0-I0-R0, dtype='float64', shape = (K, 1))
        dt = 1.0/N
        beta_return = np.zeros([K, N], dtype='float64')
        I_return = np.zeros([K, N], dtype='float64')
        R_return = np.zeros([K, N], dtype='float64')
        beta_return[:, 0] = beta0
        I_return[:, 0] = I0
        R_return[:, 0] = R0
        for i in range(N-1):
            next_beta = curr_beta + self.t_max * dt * self.model.g(curr_beta, dataset[:, i])
            beta_return[:, i+1] = next_beta.numpy()[:, 0].copy()
            # step di eulero in avanti
            next_S_nn = curr_S_nn - self.t_max * dt * curr_beta * curr_S_nn * curr_I_nn
            next_I_nn = curr_I_nn + self.t_max * dt * ( curr_beta * curr_S_nn * curr_I_nn -  self.alpha * curr_I_nn )
            next_R_nn = curr_R_nn + self.t_max * dt * self.alpha * curr_I_nn
            I_return[:, i+1] = next_I_nn.numpy()[:, 0].copy()
            R_return[:, i+1] = next_R_nn.numpy()[:, 0].copy()
            # update
            curr_S_nn = next_S_nn
            curr_beta = next_beta
            curr_I_nn = next_I_nn
            curr_R_nn = next_R_nn
        return beta_return, I_return, R_return
            
    
# TODO:
    # 1) Model è da rendere una classe base (ModelBase), la definizione di custom_loss
    #   è lasciata da definire alle classi derivate. Quindi si avrà una classe 
    #   del tipo: ModelSIR che utilizza il sir per definire la loss
    # 2) NetworkForSir deve essere indipendente dal moedllo epidemiologico scelto, quindi:
        # * rinominare la classe, e chiamarla NetworkForPandemic (o qualcosa di simile)
        # * La funzione compute_beta_I non deve usare uno step di EA sul SIR, 
        #   quindi deve utilizzare una qualche funzione definita dal membro self.model
        #   Una possibile idea è definire in ModelSIR un metodo chiamato ForwardEulerStep
    # 3) Rendere possibile la creazione di una rete con un solo input (beta)