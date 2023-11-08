from matplotlib import pyplot as plt
import numpy as np
import os

"""
def test_plot(beta0, g, f, save_plot = True, save_path = ''):
    # inizializza i beta
    t = np.linspace(0,1,N)
    y_real = np.zeros(N)
    y_nn = np.zeros(N)
    y_real[0] = beta0
    y_nn[0] = beta0
    curr_y = tf.constant([[beta0]], dtype='float64')
    # inizializza le S, I
    I_real = np.zeros(N)
    I_nn = np.zeros(N)
    I_real[0] = I0
    I_nn[0] = I0
    curr_I_nn = tf.constant([[I0]], dtype='float64')
    curr_S_nn = tf.constant([[S0]], dtype='float64')

    for i in range(N - 1):
        # check override fun-type
        Betaeq, T = tg.generate_temp_by_adri(b_ref)
        curr_temp = np.array([T(t[i])], dtype='float64')
        curr_betaeq = np.array([Betaeq(t[i])], dtype='float64')
        next_y = curr_y + dt * g(curr_y, curr_temp)
        y_nn[i + 1] = next_y.numpy()[0][0]
        y_real[i + 1] = y_real[i] + dt * f(y_real[i], curr_betaeq)

        next_S_nn = curr_S_nn - t_max * dt * curr_y * curr_S_nn * curr_I_nn
        next_I_nn = curr_I_nn + t_max * dt * curr_y * curr_S_nn * curr_I_nn  - t_max * dt * alpha * curr_I_nn
        I_nn[i + 1] = next_I_nn.numpy()[0][0]
        curr_y = next_y
        curr_S_nn = next_S_nn
        curr_I_nn = next_I_nn

    t, I_real = ig.generate_I(y_real)
    # plot beta e infetti
    fig, (ax1, ax2) = plt.subplots(2)
    fig.suptitle('Beta ed infetti con temperatura generata casualmente')
    ax1.plot(t, y_nn)
    ax1.plot(t, y_real)
    ax1.legend("beta generato","beta reale")
    ax2.plot(t, I_nn)
    ax2.plot(t, I_real)
    ax2.legend(["infetti generati", "infetti reali"])
    plt.show()
    if save_plot:
        print("Saving the plot in " + save_path + "...\n")
        if not os.path.exists(save_path):
            os.mkdir(save_path)
        filepath = save_path + "/casual_test.png";
        plt.savefig(fname=filepath)


def Dataset_plot(dataset, beta0, g, f, train_or_test, save_plot = True, save_path = ''):
    while(train_or_test != 0 & train_or_test != 1):
        print('Specify training(1) or test(0): ')
        train_or_test = input()
    n_plots = len(dataset[0,:,0])
    for p in range(n_plots):
        # inizializza i beta
        y_real = np.zeros(N)
        y_nn = np.zeros(N)
        y_real[0] = beta0
        y_nn[0] = beta0
        curr_y = tf.constant([[beta0]], dtype='float64')
        # inizializza le S, I
        I_real = np.zeros(N)
        I_nn = np.zeros(N)
        I_real[0] = I0
        I_nn[0] = I0
        curr_I_nn = tf.constant([[I0]], dtype='float64')
        curr_S_nn = tf.constant([[S0]], dtype='float64')

        for i in range(N - 1):
            curr_temp = np.array([dataset[0, p, i]], dtype='float64')
            next_y = curr_y + dt * g(curr_y, curr_temp)
            y_nn[i + 1] = next_y.numpy()[0][0]
            next_S_nn = curr_S_nn - t_max * dt * curr_y * curr_S_nn * curr_I_nn
            next_I_nn = curr_I_nn + t_max * dt * curr_y * curr_S_nn * curr_I_nn  - t_max * dt * alpha * curr_I_nn
            I_nn[i + 1] = next_I_nn.numpy()[0][0]
            curr_y = next_y
            curr_S_nn = next_S_nn
            curr_I_nn = next_I_nn
        if p % 5 == 0:
            beta_vec = dataset[1,p,:]
            t, I_real = ig.generate_I(beta_vec)
            # plot dei beta
            fig, (ax1, ax2) = plt.subplots(2)
            fig.suptitle('Beta ed infetti con temperatura n°' + str(p+1))
            ax1.plot(t, y_nn)
            ax1.plot(t, y_real)
            ax1.legend("beta generato", "beta reale")
            ax2.plot(t, I_nn)
            ax2.plot(t, I_real)
            ax2.legend(["infetti generati", "infetti reali"])
            plt.show()
            if save_plot:
                print("Saving the plot in " + save_path + "...\n")
                if not os.path.exists(save_path):
                    os.mkdir(save_path)
                if(train_or_test == 0):
                    string = "/test_plot"
                else:
                    string = "/training_plot"
                filepath = save_path + string + str(p + 1) + ".png";
                plt.savefig(fname=filepath)

"""

def plot_beta_I(beta, I, beta_nn, I_nn, set_type = '', plot_display = 1, save_plots = ''):
    K = beta.shape[0]
    N = beta.shape[1]
    t = np.linspace(0., 1., N)
    for k in range(K):
        if k % plot_display == 0:
            fig, (ax1, ax2) = plt.subplots(2)
            ax1.plot(t, beta_nn[k, :])
            ax1.plot(t, beta[k, :])
            ax1.legend(["beta rete", "beta reali"])
            ax2.plot(t, I_nn[k, :])
            ax2.plot(t, I[k, :])
            ax2.legend(["infetti rete", "infetti reali"])
            plt.suptitle(set_type+" n° "+str(k+1))  # Il parametro "y" regola l'altezza del titolo
            if len(save_plots) > 0:
                print("Saving the plots in " + save_plots + "...\n")
                path = "./" + save_plots;
                if not os.path.exists(path):
                    os.mkdir(path)
                filepath2 = path + "/infettitrain" + str(k + 1) + ".png";
                plt.savefig(fname=filepath2)
            plt.tight_layout()  # Per evitare sovrapposizioni
            plt.show()
        
    
    
