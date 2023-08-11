import numpy as np

class Perceptron:
    def __init__(self, x_len, fun, learn_coef):
        self.__fun = fun
        self.__learn_coef = learn_coef/2
        self.__weight = 2*np.random.rand(1, x_len+1)-1

    def eval(self, x_in):
        x = np.concatenate(([-1], x_in))
        return self.__fun(self.__weight @ x)

    # a epoc of training
    def trn(self, data_set, yd, method = 'gradient'):
        # para cada dato del data_set
        for i in range(len(data_set)):
            x = np.concatenate(([-1], data_set[i])) # agregamos bias
            if method == 'c_error':
                y = self.__fun(self.__weight @ x)
                self.__weight += ((0.5*self.__learn_coef)*(yd[i] - y)*x)
            else:
                e = self.__weight @ x
                self.__weight -= (2*self.__learn_coef*(yd[i] - e) * x)

    # Measures the percentage of success of the model
    def score(self, data_set, y_d):
        err_tot = 0
        for i in range(len(data_set)):
            err_tot += (y_d[i] - self.eval(data_set[i]))**2
        err = err_tot/len(data_set)
        return err 
        
