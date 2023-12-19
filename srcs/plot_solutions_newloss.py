from matplotlib import pyplot as plt
import numpy as np
import os
from estrai_dati import start_vec_val, start_vec_train, n_mesi
import math

reg_list = ['Abruzzo',
            'Basilicata',
            'Calabria',
            'Campania',
            'Emilia-Romagna',
            'Friuli Venezia Giulia',
            'Lazio',
            'Liguria',
            'Lombardia',
            'Marche',
            'Piemonte',
            'Puglia',
            'Sardegna',
            'Sicilia',
            'Toscana',
            'Umbria',
            'Veneto',
            'P.A. Bolzano',
            'P.A. Trento']


def plot_beta_I(I_nn, beta_nn, I, n_iter = [], beta=[], set_type='', save_plots='', plot = False):
    K = beta_nn.shape[0]
    N = beta_nn.shape[1]
    t = np.linspace(0., 1., N)
    for k in range(K):
        fig, (ax1, ax2) = plt.subplots(2)
        ax1.plot(t, np.absolute(beta_nn[k, :]))
        if len(beta) > 0:  # di default beta = [], vuol dire che i beta veri non sono noti
            ax1.plot(t, beta[k, :])
            ax1.legend(["beta rete", "beta reali"])
        else:
            ax1.legend(["beta rete"])
        ax2.plot(t, I_nn[k, :])
        ax2.plot(t, I[k, :])
        ax2.legend(["infetti rete", "infetti reali"])
        curr_reg = reg_list[k % 19]
        if set_type == 'train':
            curr_day = start_vec_train[math.floor(k / 19)]
        elif set_type == 'val':
            curr_day = start_vec_val[math.floor(k / 19)]
        plt.suptitle(set_type + " n° " + str(k + 1) + ': ' + curr_reg + ' dal ' + curr_day + ', ' + str(
            n_mesi) + ' mesi')  # Il parametro "y" regola l'altezza del titolo
        if len(save_plots) > 0:
            if not n_iter == []:
                save_plots = save_plots + '_iter' + str(n_iter)
            print("Saving the plots in " + save_plots + "...\n")
            path = "./" + save_plots;
            if not os.path.exists(path):
                os.mkdir(path)
            filepath2 = path + "/" + set_type + " n°" + str(k + 1) + ".png";
            plt.savefig(fname=filepath2)
        plt.tight_layout()  # Per evitare sovrapposizioni
        #plt.close()
        if plot:
            plt.show()

def plot_beta_I_alpha(I_nn, R_nn, beta_nn, alpha_nn, I, R, n_iter = [], beta=[], alpha_vec=[], set_type='', save_plots='', plot = False):
    K = beta_nn.shape[0]
    N = beta_nn.shape[1]
    t = np.linspace(0., 1., N)
    for k in range(K):
        fig, (axs) = plt.subplots(2,2, figsize = (10,5))
        axs[0,0].plot(t, np.absolute(beta_nn[k, :]))
        if len(beta) > 0:  # di default beta = [], vuol dire che i beta veri non sono noti
            axs[0,0].plot(t, beta[k, :])
            axs[0,0].legend(["beta rete", "beta reali"])
        else:
            axs[0,0].legend(["beta rete"])
        axs[1,0].plot(t, I_nn[k, :])
        axs[1,0].plot(t, I[k, :])
        axs[1,0].legend(["infetti rete", "infetti reali"])

        axs[0,1].plot(t, np.absolute(alpha_nn[k, :]))
        if len(alpha_vec) > 0:  # di default beta = [], vuol dire che i beta veri non sono noti
            axs[0,1].plot(t, alpha_vec[k, :])
            axs[0,1].legend(["alpha rete", "alpha reali"])
        else:
            axs[0,1].legend(["alpha rete"])
        axs[1,1].plot(t, R_nn[k, :])
        axs[1,1].plot(t, R[k, :])
        axs[1,1].legend(["rimossi rete", "rimossi reali"])

        curr_reg = reg_list[k % 19]
        if set_type == 'train':
            curr_day = start_vec_train[math.floor(k / 19)]
        elif set_type == 'val':
            curr_day = start_vec_val[math.floor(k / 19)]
        plt.suptitle(set_type + " n° " + str(k + 1) + ': ' + curr_reg + ' dal ' + curr_day + ', ' +
                     str(n_mesi) + ' mesi')  # Il parametro "y" regola l'altezza del titolo
        if len(save_plots) > 0:
            if not n_iter == []:
                save_plots_new = save_plots + '_iter' + str(n_iter)
            else:
                save_plots_new = save_plots
            print("Saving the plots in " + save_plots_new + "...\n")
            path = "./" + save_plots_new;
            if not os.path.exists(path):
                os.mkdir(path)
            filepath2 = path + "/" + set_type + " n°" + str(k + 1) + ".png";
            plt.savefig(fname=filepath2)
        plt.tight_layout()  # Per evitare sovrapposizioni
        plt.close()
        if plot:
            plt.show()

def plot_I(I_nn, I, n_iter = [], set_type='', plot_display=1, save_plots='', plot = False):
    K = I_nn.shape[0]
    N = I_nn.shape[1]
    t = np.linspace(0., 1., N)
    for k in range(K):
        if k % plot_display == 0:
            fig, ax = plt.subplots()
            # ax1.plot(t, np.absolute(beta_nn[k, :]))
            # if len(beta) > 0:  # di default beta = [], vuol dire che i beta veri non sono noti
            #     ax1.plot(t, beta[k, :])
            #     ax1.legend(["beta rete", "beta reali"])
            # else:
            #     ax1.legend(["beta rete"])
            ax.plot(t, I_nn[k, :])
            ax.plot(t, I[k, :])
            ax.legend(["infetti rete", "infetti reali"])
            curr_reg = reg_list[k % 19]
            if set_type == 'train':
                curr_day = start_vec_train[math.floor(k / 19)]
            elif set_type == 'val':
                curr_day = start_vec_val[math.floor(k / 19)]
            plt.suptitle(set_type + " n° " + str(k + 1) + ': ' + curr_reg + ' dal ' + curr_day + ', ' + str(
                n_mesi) + ' mesi')  # Il parametro "y" regola l'altezza del titolo
            if len(save_plots) > 0:
                if not n_iter == []:
                    save_plots_new = save_plots + '_iter' + str(n_iter)
                else:
                    save_plots_new = save_plots
                print("Saving the plots in " + save_plots_new + "...\n")
                path = "./" + save_plots_new;
                if not os.path.exists(path):
                    os.mkdir(path)
                filepath2 = path + "/" + set_type + " n°" + str(k + 1) + ".png";
                plt.savefig(fname=filepath2)
            plt.tight_layout()  # Per evitare sovrapposizioni
            #plt.close()
            if plot:
                plt.show()



