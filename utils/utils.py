import numpy as np
import matplotlib.pyplot as plt

def funcionLinealPerceptron(weight):
    weight = weight[0]
    y1 = ((-weight[1]/weight[2])*-1) + (weight[0]/weight[2])
    y2 = ((-weight[1]/weight[2])* 1) + (weight[0]/weight[2])

    plt.plot([-1, 1], [y1,y2])
    plt.show()

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