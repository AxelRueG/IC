import numpy as np

class Perceptron:
    def __init__(self, x_len, fun, learn_coef):
        self.__fun = fun
        self.__learn_coef = learn_coef/2
        self.__weight = 2*np.random.rand(1, x_len+1)-1
        # self.__weight = np.array([0.5, 0.5, 0.5])

    def eval(self, x_in):
        x = np.concatenate(([-1], x_in))
        return self.__fun(self.__weight @ x)

    def trn(self, x, d):
        y = self.eval(x)
        x_ext = np.concatenate(([-1], x))
        err = (d-y)
        self.__weight += ((self.__learn_coef)*err*x_ext)
        return err

    def trn_grad(self, x, d):
        x_ext = np.concatenate(([-1], x))
        err_inst = d - (self.__weight @ x_ext)
        self.__weight += (2*self.__learn_coef*err_inst*x_ext)
        return err_inst