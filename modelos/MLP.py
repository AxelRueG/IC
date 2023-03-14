import numpy as np

# np.random.seed(2)

class MultiLayerPreceptron():

    def __init__(self, num_in, architecture, learn_coef, fun):
        self.__learn_coef = learn_coef
        self.__architecture = architecture
        self.__fun = fun
        self.__layers = len(architecture)
        self.__weights = []

        for i in range(len(architecture)):
            # add 1 for the bias
            w = np.random.rand(self.__architecture[i], num_in+1)-0.5
            self.__weights.append(w)
            num_in = architecture[i]

    # this function add a bias to the input
    def __add_bias(self, x):
        return np.concatenate(([-1], x))

    # this function calculate all layer output and return this in a list
    def __propagation(self, x_in):
        x = x_in.copy()
        ys = []
        for i in range(self.__layers):
            x = self.__add_bias(x)
            x = self.__fun(self.__weights[i] @ x)
            ys.append(x.copy())

        return ys

    def eval(self, x_in):
        return self.__propagation(x_in)[-1]

    # this function calculate the gradient of all layes
    def __back_propagation(self, yd, d):
        dw = []
        # output layer
        di = (d-yd[-1])*(1-yd[-1])*(1+yd[-1])
        dw.append(di.copy())
        # internal layers
        for i in range(self.__layers-1, 0, -1):
            di = (np.transpose(self.__weights[i][:, 1::]) @ di) * (1-yd[i-1]) * (1+yd[i-1])
            dw.insert(0, di.copy())
        return dw

    # this function recalculate the weights
    def __amend(self, ys, delta):
        for i in range(self.__layers):
            y = self.__add_bias(ys[i])
            dw = self.__learn_coef * (np.transpose([delta[i]]) @ [y])
            self.__weights[i] += dw

    # this function trainig the model and return the absolute error
    def traing(self, x_in, d):
        ys = self.__propagation(x_in)
        dw = self.__back_propagation(ys, d)
        ys.insert(0, x_in.copy())
        self.__amend(ys, dw)

        return abs(d-ys[-1])
