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
def plot_beta_I(I_nn, beta_nn, I, R_nn = [], R = [], beta = [], set_type = '', plot_display = 1, save_plots = ''):
    K = beta_nn.shape[0]
    N = beta_nn.shape[1]
    t = np.linspace(0., 1., N)
    if not R_nn == [] and not R == []:
        num_axs = 3
    else:
        num_axs = 2
    for k in range(K):
        if k % plot_display == 0:
            fig, (axs) = plt.subplots(num_axs,1, figsize = (6,num_axs*3))
            axs[0].plot(t, beta_nn[k, :])
            if len(beta) > 0: # di default beta = [], vuol dire che i beta veri non sono noti
                axs[0].plot(t, beta[k, :])
                axs[0].legend(["beta rete", "beta reali"])
            else:
                axs[0].legend(["beta rete"])
            axs[1].plot(t, I_nn[k, :])
            axs[1].plot(t, I[k, :])
            axs[1].legend(["infetti rete", "infetti reali"])
            if num_axs == 3:
                axs[2].plot(t, R_nn[k, :])
                axs[2].plot(t, R[k, :])
                axs[2].legend(["rimossi rete", "rimossi reali"])
            curr_reg = reg_list[k % 19]
            if set_type == 'train':
                curr_day = start_vec_train[math.floor(k/19)]
            elif set_type == 'val':
                curr_day = start_vec_val[math.floor(k / 19)]
            plt.suptitle(set_type+" n° "+str(k+1) + ': ' + curr_reg + ' dal ' + curr_day + ', ' + str(n_mesi) + ' mesi')  # Il parametro "y" regola l'altezza del titolo
            if len(save_plots) > 0:
                print("Saving the plots in " + save_plots + "...\n")
                path = "./" + save_plots;
                if not os.path.exists(path):
                    os.mkdir(path)
                filepath2 = path + "/" + set_type + " n°" + str(k + 1) + ".png";
                plt.savefig(fname=filepath2)
            plt.tight_layout()  # Per evitare sovrapposizioni
            plt.show()

# def plot_beta_I_R(I_nn, R_nn, beta_nn, I, R, beta = [], set_type = '', plot_display = 1, save_plots = ''):
#     K = beta_nn.shape[0]
#     N = beta_nn.shape[1]
#     t = np.linspace(0., 1., N)
#     for k in range(K):
#         if k % plot_display == 0:
#             fig, (ax1, ax2, ax3) = plt.subplots(3)
#             ax1.plot(t, beta_nn[k, :])
#             if len(beta) > 0: # di default beta = [], vuol dire che i beta veri non sono noti
#                 ax1.plot(t, beta[k, :])
#                 ax1.legend(["beta rete", "beta reali"])
#             else:
#                 ax1.legend(["beta rete"])
#             ax2.plot(t, I_nn[k, :])
#             ax2.plot(t, I[k, :])
#             ax2.legend(["infetti rete", "infetti reali"])
#             ax3.plot(t, R_nn[k, :])
#             ax3.plot(t, R[k, :])
#             ax3.legend(["rimossi rete", "rimossi reali"])
#             curr_reg = reg_list[k % 19]
#             if set_type == 'train':
#                 curr_day = start_vec_train[math.floor(k/19)]
#             elif set_type == 'val':
#                 curr_day = start_vec_val[math.floor(k / 19)]
#             plt.suptitle(set_type+" n° "+str(k+1) + ': ' + curr_reg + ' dal ' + curr_day + ', ' + str(n_mesi) + ' mesi')  # Il parametro "y" regola l'altezza del titolo
#             if len(save_plots) > 0:
#                 print("Saving the plots in " + save_plots + "...\n")
#                 path = "./" + save_plots;
#                 if not os.path.exists(path):
#                     os.mkdir(path)
#                 filepath2 = path + "/" + set_type + " n°" + str(k + 1) + ".png";
#                 plt.savefig(fname=filepath2)
#             plt.tight_layout()  # Per evitare sovrapposizioni
#             plt.show()
        
    
    
