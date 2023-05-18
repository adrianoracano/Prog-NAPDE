# -*- coding: utf-8 -*-
"""
Created on Sun May 14 13:16:12 2023

@author: alean
"""

import MyCrankNicolsonClass as cnc

def print_help():
    print("Questa è una funzione provvisoria pergenerareun dataset contenete i beta esatti e le temperature.")
    print("La funzione prende in input: ")
    print("1) functions  :  una lista contenente le funzioni temperatura (funzioni del tempo)")
    print("2) data  :  un dict contenente i paramerti del problema")
    print("i campi di 'data' sono i seguenti:")
    print(" beta0  : condizione iniziale, ")
    print(" f  :  funzione da usare per il sistema del tipo:")
    print("     beta'(t) = f(beta(t), T(t))")
    print("  quindi f è una funzione di T e di beta,")
    print(" t_max  :  istante di tempo finale,")
    print(" N  :  numero di nodi,")
    print("\n")
    print("L'output è un np.array di tre dimensioni: dataset[i, j, k], dove:")
    print(" i vale 0 o 1: i = 0 corrisponde alle temperature, i = 1 corrisponde ai beta,")
    print(" j è l'indice per le temperature (quindi se vengono scelte J temperature j varia da 0 a J-1),")
    print(" k è l'indice per i timestep (quindi varia da 0 a N-1)\n\n")
    print("La funzione example() esegue un esempio, la funzione print_example() mostra il codice dell'esempio")
    
    
def generate_dataset(functions, data):
    #from utilities
    import MyCrankNicolsonClass as cnc
    import numpy as np
    n_functions = len(functions)
    dataset = np.zeros([2, n_functions, data["N"]])
    k = 0
    for T in functions:
        f = data["f"]
        def dbeta(beta, t):
            return f(beta, T(t))
        sys = [dbeta]
        cn_solver = cnc.CrankNicolson(sys , data["beta0"], data["t_max"], data["N"])
        cn_solver.compute_solution()
        t, beta = cn_solver.get_solution()
        for i in range(len(t)):
            dataset[0, k, i] = T( t[i] )
            dataset[1, k, i] = beta[0, i]
        k = k + 1
    return dataset.copy()


def example():
    import numpy as np
    import matplotlib.pyplot as plt
    def f(beta, T):
        return -T*beta[0]
    def T1(t):
        return 20./(1 + t**2)
    def T2(t):
        return 2*t
    functions = [T1, T2]
    data = {
        "N" : 250,
        "t_max" : 3.0,
        "beta0" : np.array([0.5]),
        "f" : f
    }
    t = np.linspace(0., data["t_max"], data["N"])
    dataset = generate_dataset(functions, data)
    fig, (ax1, ax2) = plt.subplots(2)
    fig.suptitle('Temperature e beta')
    ax1.plot(t, dataset[0, 0, ])
    ax1.plot(t, dataset[0, 1, ])
    ax1.legend(["T1(t) = 20./(1 + t**2)", "T2(t) = 2*t"])
    ax2.plot(t, dataset[1, 0, ])
    ax2.plot(t, dataset[1, 1, ])
    ax2.legend(["beta1(t)", "beta2(t)"])
    plt.show()
    
    return dataset

            
    
