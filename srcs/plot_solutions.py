from matplotlib import pyplot as plt
import numpy as np
import os
# from estrai_dati import start_vec_val, start_vec_train, n_mesi
import math
import matplotlib

beta_str = chr(946)
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


def plot_beta_I(I_nn, beta_nn, I, beta = [], set_type = '', plot_display = 1, save_plots = ''):
    K = beta_nn.shape[0]
    N = beta_nn.shape[1]
    t = np.linspace(0., 1., N)
    for k in range(K):
        if k % plot_display == 0:
            fig, (ax1, ax2) = plt.subplots(2,1, figsize = (16,9))
            ax1.plot(t, beta_nn[k, :])
            if len(beta) > 0: # di default beta = [], vuol dire che i beta veri non sono noti
                ax1.plot(t, beta[k, :])
                ax1.legend([ beta_str + " rete", beta_str + " reali"])
            else:
                ax1.legend([beta_str + " rete"])
            ax2.plot(t, I_nn[k, :])
            ax2.plot(t, I[k, :])
            ax2.legend(["infetti rete", "infetti reali"])
            plt.suptitle(set_type+" n° "+str(k))
            # ax3.plot(t, R_nn[k, :])
            # ax3.plot(t, R[k, :])
            # ax3.legend(["rimossi rete", "rimossi reali"])
            # curr_reg = reg_list[k % 19]
            # if set_type == 'train':
            #     curr_day = start_vec_train[math.floor(k/19)]
            # elif set_type == 'val':
            #     curr_day = start_vec_val[math.floor(k / 19)]
            # plt.suptitle(set_type+" n° "+str(k+1) + ': ' + curr_reg + ' dal ' + curr_day + ', ' + str(n_mesi) + ' mesi')  # Il parametro "y" regola l'altezza del titolo
            if len(save_plots) > 0:
                print("Saving the plots in " + save_plots + "...\n")
                path = "./" + save_plots;
                if not os.path.exists(path):
                    os.makedirs(path)
                filepath2 = path + "/" + set_type + " n°" + str(k + 1) + ".png";
                plt.savefig(fname=filepath2)
            plt.tight_layout()  # Per evitare sovrapposizioni
            #plt.show()
            plt.close()

def plot_beta_I_2(I_nn, beta_nn, I, beta = [], set_type = '', plot_display = 1, save_plots = '', rows = 1, cols = 5, n_giorni = None, date = None): #rows è 1 ma di fatto è 2 perchè ci sono anche infetti
    K = beta_nn.shape[0]
    N = beta_nn.shape[1]
    t = np.linspace(0., 1., N) * n_giorni
    K_vec = np.arange(start = 0, stop = K, step = plot_display * rows * cols)
    real_cols = cols
    """ if max(1, cols - np.mod(rows * cols, K - K_vec[-1])) == 1:
        avoid_last_loop = 1 #se rimane solo un'immagine fuori dà sempre problemi """
    
    for k in K_vec:
        #subplots
        if k == K_vec[-1] :
            if np.mod(K, k) == 1:
                break
            else:
                real_cols = np.mod(K, k)
                print("real_cols : " + str(real_cols))
        fig, ax = plt.subplots(nrows = rows * 2, ncols = real_cols, figsize = (real_cols * 3 + 1, rows*3))
        plt.subplots_adjust(left=0.06, bottom=0.137, right=0.98, top=0.844, wspace=0.05, hspace=0.0)
        y_lim0up = []
        y_lim0down = []
        y_lim1up = []
        y_lim1down = []
        for i in range(rows):
            for j in range(real_cols):
                ax[2*i , j].plot(t, beta_nn[k + j + i * cols, :])
                if not date == None:
                    ax[2*i , j].set_title(date[k + j + i * cols])
                if len(beta) > 0: # di default beta = [], vuol dire che i beta veri non sono noti
                    ax[2*i , j].plot(t, beta[k + j + i* cols, :])
                    if j % cols == 0:
                        ax[2*i , j].legend(["network " + beta_str ,"real " + beta_str ],  bbox_to_anchor=(-0.2, 1))#, prop={'size': 8})
                    #ax[2*i , j].set_xlabel("day")
                        #ax[2*i , j].set_ylabel(beta_str)
                    else:
                        #print("removing y_ticks from beta")
                        ax[2*i , j].set_yticks([])
                else:
                    if j % cols == 0:
                        ax[2*i, j].legend([beta_str + " rete"], bbox_to_anchor=(-0.23, 1))#, prop={'size': 8})
                        ax[2*i , j].set_xlabel("day")
                        #ax[2*i , j].set_ylabel(beta_str)
                ax[2*i , j].set_xticks([])
                ax[2*i + 1 , j].plot(t, I_nn[k + j + i * cols, :])
                ax[2*i + 1 , j].plot(t, I[k + j + i * cols, :])
                #ax[2*i + 1 , j].set_xticks([range(min(np.round(t)), max(np.round(t)) + 1)])
                if j % cols == 0:
                    ax[2*i + 1 , j].legend(["network I", "real I"], bbox_to_anchor=(-0.23, 1))#, prop={'size': 8})
                    #ax[2*i + 1, j].set_ylabel("Infetti")
                else:
                    #print("removing y_ticks from I")
                    ax[2*i + 1, j].set_yticks([])
                ax[2*i + 1, j].set_xlabel("day")

                #ax[2*i , j].set_title(set_type + ' n°' + str(k + j + i * cols + 1))
                #if i != 0 or j != 0:
                 #   ax[2*i, j].set_ylim(ax[0,0].get_ylim())
                  #  ax[2*i + 1, j].set_ylim(ax[1,0].get_ylim())

                """ if not n_giorni == None:
                    ax[2*i + 1, j].set_xticks(np.arange(1, n_giorni, 3)) """
                plt.subplots_adjust(left=0.155, bottom=0.165, right=0.98, top=0.75, wspace=0, hspace=0.0)
    #plt.tight_layout()  # Per evitare sovrapposizioni
                y_lim0down.append(ax[2*i , j].get_ylim()[0])
                y_lim0up.append(ax[2*i , j].get_ylim()[1])
                y_lim1down.append(ax[2*i + 1 , j].get_ylim()[0])
                y_lim1up.append(ax[2*i + 1, j].get_ylim()[1])
            #ax[2*i, j].set_title(set_type + " n° " + str(k + j + i * cols))
        height0 = np.max(y_lim0up) - np.min(y_lim0down)
        height1 = np.max(y_lim1up) - np.min(y_lim1down)
        fig.suptitle(set_type + " n°" + str(k + 1))
        for i in range(rows):
            for j in range(real_cols):
                ax[2*i , j].set_ylim(np.min(y_lim0down) - 0.1 * height0, np.max(y_lim0up) + 0.1 * height0)
                ax[2*i + 1 , j].set_ylim(np.min(y_lim1down) - 0.1 * height1, np.max(y_lim1up) + 0.1 * height1)
        if len(save_plots) > 0:
            print("Saving the plots in " + save_plots + "...\n")
            path = "./" + save_plots;
            if not os.path.exists(path):
                os.makedirs(path)
            filepath2 = path + "/" + set_type + " n°" + str(k + 1) + ".png";
            plt.savefig(fname=filepath2)
    plt.show()
    #plt.close()



