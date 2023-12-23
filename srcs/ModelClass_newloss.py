import tensorflow as tf
from tensorflow import keras as tfk
from keras import layers as tfkl
import numpy as np
import pickle

tf.keras.backend.set_floatx('float64')


class Model:
    def __init__(self, n_input, n_hidden, learning_rate, b_ref, addDropout=False, \
                 addBNorm=True, load_path='', MSE_loss = True, NormalizeInput=False, use_keras=False, use_dI = False, no_SIR = False, learn_alpha = False, num_layers = 1):

        self.n_input = n_input
        self.n_hidden = n_hidden
        self.b_ref = b_ref
        self.use_dI = use_dI
        self.use_keras = use_keras
        self.n_iter = 0
        self.loss = self.custom_loss

        ########################
        # training without SIR
        #######################
        if no_SIR:
            self.loss = self.custom_loss_I

        ###############
        # learn alpha
        ###############

        if learn_alpha:
            self.n_param = 2
            self.n_output = 2
            if MSE_loss:
                if use_dI:
                    self.loss = self.custom_loss_alpha_MSE_dI
                else:
                    self.loss = self.custom_loss_alpha_MSE
            else:
                if use_dI:
                    self.loss = self.custom_loss_alpha_abs_dI
                else:
                    self.loss = self.custom_loss_alpha_abs
        else:
            self.n_param = 1
            self.n_output = 1
            self.loss = self.custom_loss

        #############
        # use keras
        #############

        self.use_keras = use_keras
        if self.use_keras:
            self.train_step = self.train_step_keras
            if NormalizeInput:
                self.g = self.g_normalized_keras
            else:
                self.g = self.g_not_normalized_keras
        else:
            self.train_step = self.train_step_std
            if not NormalizeInput:
                self.g = self.g_normalized_std
            else:
                self.g = self.g_not_normalized_std

        ###############
        # Keras model
        ###############
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
        if addBNorm:
            self.keras_model.add(tfkl.BatchNormalization())
        for i in range(num_layers - 1):
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
            if addBNorm:
                self.keras_model.add(tfkl.BatchNormalization())
            self.keras_model.add(tfkl.Activation('sigmoid'))
        if addDropout:
            self.keras_model.add(tfkl.Dropout(0.3))
        self.keras_model.add(tfkl.Dense(1))
        self.keras_model.add(tfkl.BatchNormalization())

        ##################################
        # loading or generating new weights
        ##################################

        if (len(load_path) > 0):
            print("Loading model from " + load_path + "...\n")
            self.load_model(load_path)
        else:
            self.weights = {
                'h1': tf.Variable(tf.random.normal([self.n_input, n_hidden], dtype='float64'), dtype='float64'),
                # 'h2': tf.Variable(tf.random.normal([self.n_input, n_hidden], dtype='float64'), dtype='float64'),
                'out': tf.Variable(tf.random.normal([n_hidden, self.n_output], dtype='float64'), dtype='float64')
            }
            self.biases = {
                'b1': tf.Variable(tf.random.normal([n_hidden], dtype='float64'), dtype='float64'),
                # 'b2': tf.Variable(tf.random.normal([n_hidden], dtype='float64'), dtype='float64'),
                'out': tf.Variable(tf.random.normal([self.n_output], dtype='float64'), dtype='float64')
            }
            print("Generating new model with standard parameters ...\n")

    #####################
    # Model class methods
    #####################

    def multilayer_perceptron(self, x):
        layer_1 = tf.add(tf.matmul(x, self.weights['h1']), self.biases['b1'])
        layer_1 = tf.nn.sigmoid(layer_1)
        # layer_2 = tf.add(tf.matmul(x, self.weights['h2']), self.biases['b2'])
        # layer_2 = tf.nn.sigmoid(layer_2)
        output = tf.matmul(layer_1, self.weights['out']) + self.biases['out']
        return output

    def g_not_normalized_keras(self, y, v):
        v.shape = (v.shape[0], self.n_input - self.n_param)
        tv = tf.constant(v, dtype='float64')
        x = tf.concat([y, tv], 1)
        return self.keras_model(x)

    def g_not_normalized_std(self, y, v):
        v.shape = (v.shape[0], self.n_input - self.n_param)
        tv = tf.constant(v, dtype='float64')
        x = tf.concat([y, tv], 1)
        return self.multilayer_perceptron(x)

    def g_normalized_keras(self, y, v):
        y = (1.0 / self.b_ref) * y
        v = (1.0 / self.b_ref) * v
        v.shape = (v.shape[0], self.n_input - self.n_param)
        tv = tf.constant(v, dtype='float64')
        x = tf.concat([y, tv], 1)
        return self.keras_model(x)

    def g_normalized_std(self, y, v):
        y = (1.0 / self.b_ref) * y
        v = (1.0 / self.b_ref) * v
        v.shape = (v.shape[0], self.n_input - self.n_param)
        tv = tf.constant(v, dtype='float64')
        x = tf.concat([y, tv], 1)
        return self.multilayer_perceptron(x)

    def custom_loss(self, dataset, I, R, t_max, alpha, beta0):
        summation = []
        K = I.shape[0]
        N = I.shape[1]
        b = np.zeros([K, 1], dtype='float64')
        b[:, 0] = beta0  # potrebbe dare problemi?
        curr_beta = tf.constant(b, dtype='float64')
        # curr_I_nn = tf.constant(sir_0[1] * np.ones([K, 1], dtype='float64'), dtype='float64')
        curr_I_nn = tf.constant(I[:, 0], dtype='float64', shape=(K, 1))
        # curr_S_nn = tf.constant(sir_0[0] * np.ones([K, 1], dtype='float64'), dtype='float64')
        curr_S_nn = tf.constant(1.0 - I[:, 0], dtype='float64', shape=(K, 1))
        dt = 1.0 / N
        for i in range(N - 1):
            curr_beta = tf.abs(curr_beta)
            next_beta = curr_beta + t_max * dt * self.g(curr_beta, dataset[:, i])
            # step di eulero in avanti
            next_S_nn = curr_S_nn - t_max * dt * curr_beta * curr_S_nn * curr_I_nn
            next_I_nn = curr_I_nn + t_max * dt * (curr_beta * curr_S_nn * curr_I_nn - alpha * curr_I_nn)
            I_exact = I[:, i + 1]
            dI_exact = (I[:, i + 1] - I[:, i]) / dt
            dI_nn = t_max * (curr_beta * curr_S_nn * curr_I_nn - alpha * curr_I_nn)
            I_exact.shape = (I_exact.shape[0], 1)
            dI_exact.shape = (dI_exact.shape[0], 1)
            summation.append(tf.reduce_mean(tf.abs(next_I_nn - I_exact)))
            if self.use_dI:
                summation.append(tf.reduce_mean(tf.abs(dI_exact - dI_nn)))
            curr_beta = next_beta
            curr_S_nn = next_S_nn
            curr_I_nn = next_I_nn
        return tf.reduce_sum(summation)

    def custom_loss_alpha_MSE(self, dataset, I, R, t_max, alpha, beta0):
        summation = []
        K = I.shape[0]
        N = I.shape[1]
        param = np.zeros([K, 2], dtype='float64')
        param[:, 0] = beta0  # potrebbe dare problemi?
        param[:, 1] = alpha
        curr_param = tf.constant(param, dtype='float64')
        curr_I_nn = tf.constant(I[:, 0], dtype='float64', shape=(K, 1))
        curr_S_nn = tf.constant(1.0 - I[:, 0] - R[:,0], dtype='float64', shape=(K, 1))
        curr_R_nn = tf.constant(R[:, 0], dtype='float64', shape=(K, 1))
        dt = 1.0 / N
        m_alpha = tf.constant([[0], [1]], dtype='float64')
        m_beta = tf.constant([[1], [0]], dtype='float64')
        for i in range(N - 1):
            next_param = curr_param + t_max * dt * self.g(curr_param, dataset[:, i])
            # step di eulero in avanti
            next_S_nn = curr_S_nn - t_max * dt * tf.tensordot(curr_param, m_beta, axes = 1) * curr_S_nn * curr_I_nn
            next_I_nn = curr_I_nn + t_max * dt * (tf.tensordot(curr_param, m_beta, axes = 1) * curr_S_nn * curr_I_nn - tf.tensordot(curr_param, m_alpha, axes = 1) * curr_I_nn)
            next_R_nn = curr_R_nn + t_max * dt * tf.tensordot(curr_param, m_alpha, axes = 1) * curr_I_nn
            I_exact = I[:, i + 1]
            dI_exact = (I[:, i + 1] - I[:, i]) / dt
            R_exact = R[:, i + 1]
            dR_exact = (R[:, i + 1] - R[:, i]) / dt
            I_exact.shape = (I_exact.shape[0], 1)
            dI_exact.shape = (dI_exact.shape[0], 1)
            R_exact.shape = (R_exact.shape[0], 1)
            dR_exact.shape = (dR_exact.shape[0], 1)
            diff_I_l2 = tf.math.square(tf.math.add(next_I_nn, - I_exact))
            diff_R_l2 = tf.math.square(tf.math.add(next_R_nn, - R_exact))
            diff = tfkl.concatenate((diff_I_l2, diff_R_l2), axis=1)
            summation.append(tf.reduce_mean(diff))
            curr_param = next_param
            curr_S_nn = next_S_nn
            curr_I_nn = next_I_nn
            curr_R_nn = next_R_nn
        return tf.reduce_sum(summation)

    def custom_loss_alpha_MSE_dI(self, dataset, I, R, t_max, alpha, beta0):
        summation = []
        K = I.shape[0]
        N = I.shape[1]
        param = np.zeros([K, 2], dtype='float64')
        param[:, 0] = beta0  # potrebbe dare problemi?
        param[:, 1] = alpha
        curr_param = tf.constant(param, dtype='float64')
        curr_I_nn = tf.constant(I[:, 0], dtype='float64', shape=(K, 1))
        curr_S_nn = tf.constant(1.0 - I[:, 0] - R[:,0], dtype='float64', shape=(K, 1))
        curr_R_nn = tf.constant(R[:, 0], dtype='float64', shape=(K, 1))
        dt = 1.0 / N
        m_alpha = tf.constant([[0], [1]], dtype='float64')
        m_beta = tf.constant([[1], [0]], dtype='float64')
        for i in range(N - 1):
            next_param = curr_param + t_max * dt * self.g(curr_param, dataset[:, i])
            # step di eulero in avanti
            next_S_nn = curr_S_nn - t_max * dt * tf.tensordot(curr_param, m_beta, axes = 1) * curr_S_nn * curr_I_nn
            next_I_nn = curr_I_nn + t_max * dt * (tf.tensordot(curr_param, m_beta, axes = 1) * curr_S_nn * curr_I_nn - tf.tensordot(curr_param, m_alpha, axes = 1) * curr_I_nn)
            next_R_nn = curr_R_nn + t_max * dt * tf.tensordot(curr_param, m_alpha, axes = 1) * curr_I_nn
            I_exact = I[:, i + 1]
            dI_exact = (I[:, i + 1] - I[:, i]) / dt
            R_exact = R[:, i + 1]
            dR_exact = (R[:, i + 1] - R[:, i]) / dt
            I_exact.shape = (I_exact.shape[0], 1)
            dI_exact.shape = (dI_exact.shape[0], 1)
            R_exact.shape = (R_exact.shape[0], 1)
            dR_exact.shape = (dR_exact.shape[0], 1)
            dI_nn = (next_I_nn - curr_I_nn)/dt
            dR_nn = (next_R_nn - curr_R_nn)/dt
            diff_I_l2 = tf.math.square(tf.math.add(next_I_nn, - I_exact))
            diff_R_l2 = tf.math.square(tf.math.add(next_R_nn, - R_exact))
            diff_dI_l2 = tf.math.square(tf.math.add(dI_nn, - dI_exact))
            diff_dR_l2 = tf.math.square(tf.math.add(dR_nn, - dR_exact))
            diff = tfkl.concatenate((diff_I_l2, diff_R_l2, diff_dI_l2, diff_dR_l2), axis=1)
            summation.append(tf.reduce_mean(diff))
            curr_param = next_param
            curr_S_nn = next_S_nn
            curr_I_nn = next_I_nn
            curr_R_nn = next_R_nn
        return tf.reduce_sum(summation)

    def custom_loss_alpha_abs(self, dataset, I, R, t_max, alpha, beta0):
        summation = []
        K = I.shape[0]
        N = I.shape[1]
        param = np.zeros([K, 2], dtype='float64')
        param[:, 0] = beta0  # potrebbe dare problemi?
        param[:, 1] = alpha
        curr_param = tf.constant(param, dtype='float64')
        curr_I_nn = tf.constant(I[:, 0], dtype='float64', shape=(K, 1))
        curr_S_nn = tf.constant(1.0 - I[:, 0] - R[:,0], dtype='float64', shape=(K, 1))
        curr_R_nn = tf.constant(R[:, 0], dtype='float64', shape=(K, 1))
        dt = 1.0 / N
        m_alpha = tf.constant([[0], [1]], dtype='float64')
        m_beta = tf.constant([[1], [0]], dtype='float64')
        for i in range(N - 1):
            next_param = curr_param + t_max * dt * self.g(curr_param, dataset[:, i])
            # step di eulero in avanti
            next_S_nn = curr_S_nn - t_max * dt * tf.tensordot(curr_param, m_beta, axes = 1) * curr_S_nn * curr_I_nn
            next_I_nn = curr_I_nn + t_max * dt * (tf.tensordot(curr_param, m_beta, axes = 1) * curr_S_nn * curr_I_nn - tf.tensordot(curr_param, m_alpha, axes = 1) * curr_I_nn)
            next_R_nn = curr_R_nn + t_max * dt * tf.tensordot(curr_param, m_alpha, axes = 1) * curr_I_nn
            I_exact = I[:, i + 1]
            dI_exact = (I[:, i + 1] - I[:, i]) / dt
            R_exact = R[:, i + 1]
            dR_exact = (R[:, i + 1] - R[:, i]) / dt
            I_exact.shape = (I_exact.shape[0], 1)
            dI_exact.shape = (dI_exact.shape[0], 1)
            R_exact.shape = (R_exact.shape[0], 1)
            dR_exact.shape = (dR_exact.shape[0], 1)
            summation.append(tf.reduce_mean(tf.abs(next_I_nn - I_exact)))
            summation.append(tf.reduce_mean(tf.abs(next_R_nn - R_exact)))
            curr_param = next_param
            curr_S_nn = next_S_nn
            curr_I_nn = next_I_nn
            curr_R_nn = next_R_nn
        return tf.reduce_sum(summation)

    def custom_loss_alpha_abs_dI(self, dataset, I, R, t_max, alpha, beta0):
        summation = []
        K = I.shape[0]
        N = I.shape[1]
        param = np.zeros([K, 2], dtype='float64')
        param[:, 0] = beta0  # potrebbe dare problemi?
        param[:, 1] = alpha
        curr_param = tf.constant(param, dtype='float64')
        # curr_I_nn = tf.constant(sir_0[1] * np.ones([K, 1], dtype='float64'), dtype='float64')
        curr_I_nn = tf.constant(I[:, 0], dtype='float64', shape=(K, 1))
        # curr_S_nn = tf.constant(sir_0[0] * np.ones([K, 1], dtype='float64'), dtype='float64')
        curr_S_nn = tf.constant(1.0 - I[:, 0] - R[:,0], dtype='float64', shape=(K, 1))
        curr_R_nn = tf.constant(R[:, 0], dtype='float64', shape=(K, 1))
        dt = 1.0 / N
        m_alpha = tf.constant([[0], [1]], dtype='float64')
        m_beta = tf.constant([[1], [0]], dtype='float64')
        for i in range(N - 1):
            # curr_param = tf.abs(curr_param)
            next_param = curr_param + t_max * dt * self.g(curr_param, dataset[:, i])
            # step di eulero in avanti
            next_S_nn = curr_S_nn - t_max * dt * tf.tensordot(curr_param, m_beta, axes = 1) * curr_S_nn * curr_I_nn
            next_I_nn = curr_I_nn + t_max * dt * (tf.tensordot(curr_param, m_beta, axes = 1) * curr_S_nn * curr_I_nn - tf.tensordot(curr_param, m_alpha, axes = 1) * curr_I_nn)
            next_R_nn = curr_R_nn + t_max * dt * tf.tensordot(curr_param, m_alpha, axes = 1) * curr_I_nn
            I_exact = I[:, i + 1]
            dI_exact = (I[:, i + 1] - I[:, i]) / dt
            R_exact = R[:, i + 1]
            dR_exact = (R[:, i + 1] - R[:, i]) / dt
            dI_nn = t_max * (tf.tensordot(curr_param, m_beta, axes = 1) * curr_S_nn * curr_I_nn - tf.tensordot(curr_param, m_alpha, axes = 1) * curr_I_nn)
            dR_nn = t_max * dt * tf.tensordot(curr_param, m_alpha, axes = 1) * curr_I_nn
            I_exact.shape = (I_exact.shape[0], 1)
            dI_exact.shape = (dI_exact.shape[0], 1)
            R_exact.shape = (R_exact.shape[0], 1)
            dR_exact.shape = (dR_exact.shape[0], 1)
            diff_I_abs = tf.math.square(tf.math.add(next_I_nn, - I_exact))
            diff_R_abs = tf.math.square(tf.math.add(next_R_nn, - R_exact))
            diff_dI_abs = tf.math.square(tf.math.add(dI_nn, - dI_exact))
            diff_dR_abs = tf.math.square(tf.math.add(dR_nn, - dR_exact))
            diff = tfkl.concatenate((diff_I_abs, diff_R_abs, diff_dI_abs, diff_dR_abs), axis=1)
            summation.append(tf.reduce_mean(diff))
            curr_param = next_param
            curr_S_nn = next_S_nn
            curr_I_nn = next_I_nn
            curr_R_nn = next_R_nn
        return tf.reduce_sum(summation)


    def custom_loss_I(self, dataset, I, R, t_max, alpha, beta0):
        summation = []
        K = I.shape[0]
        N = I.shape[1]
        curr_I_nn = tf.constant(I[:, 0], dtype='float64', shape=(K, 1))
        dt = 1.0 / N
        for i in range(N - 1):
            next_I_nn = curr_I_nn + t_max * dt * self.g(curr_I_nn, dataset[:, i])
            I_exact = I[:, i + 1]
            dI_exact = (I[:, i + 1] - I[:, i]) / dt
            dI_nn = t_max * self.g(curr_I_nn, dataset[:, i])
            I_exact.shape = (I_exact.shape[0], 1)
            dI_exact.shape = (dI_exact.shape[0], 1)
            summation.append(tf.reduce_mean(tf.abs(next_I_nn - I_exact)) + tf.reduce_mean(tf.abs(dI_exact - dI_nn)))
            curr_I_nn = next_I_nn
        return tf.reduce_sum(summation)

    def train_step_keras(self, dataset, I, R, t_max, alpha, beta0):
        with tf.GradientTape() as tape:
            loss = self.loss(dataset, I, R, t_max, alpha, beta0)
        trainable_variables = self.keras_model.trainable_variables
        gradients = tape.gradient(loss, trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, trainable_variables))
        self.n_iter = self.n_iter + 1

    def train_step_std(self, dataset, I, R, t_max, alpha, beta0):
        with tf.GradientTape() as tape:
            loss = self.loss(dataset, I, R, t_max, alpha, beta0)
        trainable_variables = list(self.weights.values()) + list(self.biases.values())
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
        #
        if self.use_keras:
            self.keras_model.save(save_path + "iter" + str(self.n_iter))
        else:
            save_path = save_path + '_iter' + str(self.n_iter)
            with open(save_path, 'wb') as file:
                pickle.dump((self.weights, self.biases, self.n_iter), file)

class NetworkForSIR:
    def __init__(self, model, display_step, t_max, alpha):
        self.model = model
        self.display_step = display_step
        self.t_max = t_max
        self.alpha = alpha
        self.default_save_path = ''
        if self.model.use_keras:
            self.default_save_path = 'saved-models/'
        else:
            self.default_save_path = 'saved-weights/'
    def train(self, dataset, I, R, val_set, I_val, R_val, beta0_train, beta0_val, max_iter, display_weights, save_path = '', saving_step = 200, validate=True):
        display_step = self.display_step
        print("Starting the training...\n")
        loss_history = np.zeros(int(max_iter / display_step))
        i_history = 0
        # if args.validate:
        loss_history_val = np.zeros(int(max_iter / display_step))
        try:
            for i in range(max_iter):
                self.model.train_step(dataset, I, R, self.t_max, self.alpha, beta0_train)
                if i % display_step == 0:
                    print("iterazione %i:" % i)
                    loss_history[i_history] = self.model.loss(dataset, I, R,\
                                                                     self.t_max, self.alpha, beta0_train)
                    print("loss on training set: %f " % loss_history[i_history])
                    #if validate:
                    loss_history_val[i_history] = self.model.loss(val_set, I_val, R_val,\
                                                                         self.t_max, self.alpha, beta0_val)
                    print("loss on validation set: %f" % loss_history_val[i_history])
                    i_history = i_history + 1
                """
                if i % display_weights == 0:
                    print("pesi all'iterazione %i:")
                    print(self.model.trainable_variables)
                self.n_iter = i
                """
                if not saving_step == None and i % saving_step == 0 and not i == 0 :
                    if save_path == '':
                        save_path = self.default_save_path
                    save_path_new = save_path
                    self.model.save_model(save_path_new)

        except KeyboardInterrupt:
            print('\nTraining interrupted by user. Proceeding to save the weights and plot the solutions...\n')
        return loss_history, loss_history_val, i_history

    # volendo si possono aggiungere qui metodi per fare plot e cose simili
    def compute_beta_I(self, dataset, I0, beta0):
        N = dataset.shape[1]
        K = dataset.shape[0]
        b = np.zeros([K, 1], dtype='float64')
        b[:, 0] = beta0  # potrebbe dare problemi?
        curr_beta = tf.constant(b, dtype='float64')
        curr_I_nn = tf.constant(I0, dtype='float64', shape=(K, 1))
        curr_S_nn = tf.constant(1.0 - I0, dtype='float64', shape=(K, 1))
        dt = 1.0 / N
        beta_return = np.zeros([K, N], dtype='float64')
        I_return = np.zeros([K, N], dtype='float64')
        beta_return[:, 0] = beta0
        I_return[:, 0] = I0
        for i in range(N - 1):
            curr_beta = tf.abs(curr_beta)
            next_beta = curr_beta + self.t_max * dt * self.model.g(curr_beta, dataset[:, i])
            beta_return[:, i + 1] = next_beta.numpy()[:, 0].copy()
            # step di eulero in avanti
            next_S_nn = curr_S_nn - self.t_max * dt * curr_beta * curr_S_nn * curr_I_nn
            next_I_nn = curr_I_nn + self.t_max * dt * (curr_beta * curr_S_nn * curr_I_nn - self.alpha * curr_I_nn)
            I_return[:, i + 1] = next_I_nn.numpy()[:, 0].copy()
            # update
            curr_S_nn = next_S_nn
            curr_beta = next_beta
            curr_I_nn = next_I_nn
        return beta_return, I_return

    def compute_beta_I_alpha(self, dataset, I0, R0, beta0, alpha):
        N = dataset.shape[1]
        K = dataset.shape[0]
        param = np.zeros([K, 2], dtype='float64')
        param[:, 0] = beta0  # potrebbe dare problemi?
        param[:,1] = alpha
        curr_param = tf.constant(param, dtype='float64')
        curr_I_nn = tf.constant(I0, dtype='float64', shape=(K, 1))
        curr_S_nn = tf.constant(1.0 - I0 - R0, dtype='float64', shape=(K, 1))
        curr_R_nn = tf.constant(R0, dtype='float64', shape=(K, 1))
        dt = 1.0 / N
        param_return = np.zeros([K, N, 2], dtype='float64')
        I_return = np.zeros([K, N], dtype='float64')
        R_return = np.zeros([K, N], dtype='float64')
        param_return[:, 0, 0] = beta0
        param_return[:, 0, 1] = alpha
        I_return[:, 0] = I0
        R_return[:,0] = R0
        for i in range(N - 1):
            #curr_param = tf.abs(curr_param)
            next_param = curr_param + self.t_max * dt * self.model.g(curr_param, dataset[:, i])
            param_return[:, i + 1, :] = next_param.numpy().copy()
            # step di eulero in avanti
            alpha_vec = np.array(curr_param[:,1])
            alpha_vec.shape = (alpha_vec.shape[0], 1)
            beta_vec = np.array(curr_param[:, 0])
            beta_vec.shape = (beta_vec.shape[0], 1)
            next_S_nn = curr_S_nn - self.t_max * dt * beta_vec * curr_S_nn * curr_I_nn
            next_I_nn = curr_I_nn + self.t_max * dt * (beta_vec * curr_S_nn * curr_I_nn - alpha_vec * curr_I_nn)
            next_R_nn = curr_R_nn + self.t_max * dt * alpha_vec * curr_I_nn
            I_return[:, i + 1] = next_I_nn.numpy()[:, 0].copy()
            R_return[:, i + 1] = next_R_nn.numpy()[:, 0].copy()
            # update
            curr_S_nn = next_S_nn
            curr_param = next_param
            curr_I_nn = next_I_nn
            curr_R_nn = next_R_nn
        beta_return = param_return[:,:,0]
        alpha_return = param_return[:,:,1]
        return beta_return, alpha_return, I_return, R_return

    def compute_I(self, dataset, I0):
        N = dataset.shape[1]
        K = dataset.shape[0]
        b = np.zeros([K, 1], dtype='float64')
        curr_I_nn = tf.constant(I0, dtype='float64', shape=(K, 1))
        dt = 1.0 / N
        I_return = np.zeros([K, N], dtype='float64')
        I_return[:, 0] = I0
        for i in range(N - 1):
            # step di eulero in avanti
            next_I_nn = curr_I_nn + self.t_max * dt * self.model.g(curr_I_nn, dataset[:, i])
            I_return[:, i + 1] = next_I_nn.numpy()[:, 0].copy()
            # update
            curr_I_nn = next_I_nn
        return I_return



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


