import numpy as np
import matplotlib.pyplot as plt

def funcionLinealPerceptron(weight):
    weight = weight[0]
    y1 = ((-weight[1]/weight[2])*-1) + (weight[0]/weight[2])
    y2 = ((-weight[1]/weight[2])* 1) + (weight[0]/weight[2])

    plt.plot([-1, 1], [y1,y2])
    plt.show()