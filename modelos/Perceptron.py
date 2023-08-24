import numpy as np

class Perceptron:
    def __init__(self, x_len, fun, learn_coef):
        self.__fun = fun
        self.__learn_coef = learn_coef
        self.__weight = 2*np.random.rand(1, x_len+1)-1

    def eval(self, x_in):
        x = np.concatenate(([-1], x_in))
        return self.__fun(self.__weight @ x)

    # El entrenamiento se realiza de a "lotes"
    def trn(self, data_set, yd, method = 'gradient'):
        # para cada dato del data_set
        for i in range(len(data_set)):
            x = np.concatenate(([-1], data_set[i])) # agregamos bias

            if method == 'c_error':
                y = self.__fun(self.__weight @ x)
                self.__weight += ((0.5*self.__learn_coef)*(yd[i] - y)*x)
            elif method == 'gradient':
                e = yd[i] - (self.__weight @ x)
                self.__weight += (2 * self.__learn_coef * e * x)
            else:
                raise ValueError("metodo no valido")


    # porcentaje de acierto del modelo
    def score(self, data_set, y_d, method="porcentaje_error"):
        err_tot = 0
        for i in range(len(data_set)):
            if method == "error_cuadratico":
                err_tot += (y_d[i] - self.eval(data_set[i]))**2
            elif method=="porcentaje_error": 
                if y_d[i] != self.eval(data_set[i]): err_tot+=1
            else:
                raise ValueError("metodo no valido")
            
        err = err_tot/len(data_set)

        return err 
    
    def getWeight(self):
        return self.__weight.copy()
        
