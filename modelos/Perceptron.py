import numpy as np

class Perceptron:
    def __init__(self, x_len, fun, learn_coef):
        self.__fun = fun
        self.__learn_coef = learn_coef
        self.__weight = 2*np.random.rand(1, x_len+1)-1

    def eval(self, x_in):
        x = np.concatenate(([-1], x_in))
        return self.__fun(self.__weight @ x)

    # a epoc of training
    def trn(self, data_set, yd, method = 'gradient'):
        x = -1 * np.ones([data_set.shape[0], data_set.shape[1]+1]) # agregamos el bias
        x[:,1:] = data_set

        # para cada dato del data_set
        for i in range(len(data_set)):
            if method == 'c_error':
                y = self.__fun(self.__weight @ x[i])
                self.__weight += ((0.5*self.__learn_coef)*(yd[i] - y)*x[i])
            elif method == 'gradient':
                e = yd[i] - (self.__weight @ x[i])
                self.__weight += (2 * self.__learn_coef* e * x[i])
            else:
                raise ValueError("metodo no valido")


    # porcentaje de aciertos
    def score(self, data_set, y_d, method="failure_rate"):
        err_tot = 0
        for i in range(len(data_set)):
            if method == "quad_error":
                err_tot += (y_d[i] - self.eval(data_set[i]))**2
            elif method=="failure_rate": 
                if y_d[i] != self.eval(data_set[i]): err_tot+=1
            else:
                raise ValueError("metodo no valido")
            
        err = err_tot/len(data_set)

        return err 
    
    def getWeight(self):
        return self.__weight.copy()
        
