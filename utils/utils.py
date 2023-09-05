import numpy as np
import matplotlib.pyplot as plt
# from tkinter import *

def funcionLinealPerceptron(weight):
    x = [-2 , 2]
    weight = weight[0]
    y1 = ((-weight[1]/weight[2])*x[0]) + (weight[0]/weight[2])
    y2 = ((-weight[1]/weight[2])*x[1]) + (weight[0]/weight[2])

    # implt.figure()
    plt.xlim([-2,2])
    plt.ylim([-2,2])
    plt.plot(x, [y1,y2])
    # plt.show()

def train_test_split(x, yd, percent):
    # mix the data to separate training and test data
    rgn = np.random.default_rng()
    it_random = rgn.permutation(np.arange(len(x)))
    cant_trn = int(len(x)*percent)
    x_trn =  x[it_random[ 0:cant_trn ],:].copy()
    x_tst =  x[it_random[ cant_trn:  ],:].copy()
    y_trn = yd[it_random[ 0:cant_trn ],:].copy()
    y_tst = yd[it_random[ cant_trn:  ],:].copy()

    return x_trn, x_tst, y_trn, y_tst

# Return two matrix with index permutated of dataset
def hold_out(num_part, data_len, percent=0.8, save= False, file_name="index_perm"):
    rgn = np.random.default_rng()
    cant_trn = int(data_len*percent)

    trn = np.zeros([cant_trn, num_part], dtype=int)
    tst = np.zeros([data_len-cant_trn, num_part], dtype=int)
    for i in range(num_part):
        it_random = rgn.permutation(data_len)
        trn[:,i] = it_random[0:cant_trn]
        tst[:,i] = it_random[cant_trn::]

    if save:
        np.savetxt(f"assets/{file_name}_trn.csv",trn,delimiter=',')
        np.savetxt(f"assets/{file_name}_tst.csv",tst,delimiter=',')

    return trn, tst

def winner_take_all(x):
    i_max = np.argmax(x)
    y = np.zeros_like(x)
    y[i_max] = 1
    for i in range(len(x)):
        if i != i_max:
            y[i] = -1
    return y

   