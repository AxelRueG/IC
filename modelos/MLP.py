import numpy as np
from utils.funciones_de_activacion import sigmoide, sgn_vec
from utils.utils import winner_take_all
# np.random.seed(2)


class MultiLayerPreceptron():

    def __init__(self, num_in: int, architecture: list, learn_coef: float, fun=sigmoide):
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

    # agrega el bias a una entrada
    def __add_bias(self, x: np.ndarray) -> np.ndarray:
        return np.concatenate(([-1], x))

    # this function calculate all layer output and return this in a list
    def __propagation(self, x_in: np.ndarray) -> list:
        x = x_in.copy()
        ys = []
        for i in range(self.__layers):
            x = self.__add_bias(x)
            x = self.__fun(self.__weights[i] @ x)
            ys.append(x.copy())

        return ys

    def eval(self, x_in: np.ndarray):
        return self.__propagation(x_in)[-1]

    # retopropagacion que calcula el gradiente
    def __back_propagation(self, yd: np.ndarray, d: np.ndarray) -> list:
        dw = []
        # output layer
        di = 0.5 * (d-yd[-1]) * (1-yd[-1]) * (1+yd[-1])
        dw.append(di.copy())
        # hidden layers
        for i in range(self.__layers-1, 0, -1):
            di = 0.5 * (np.transpose(self.__weights[i][:, 1::]) @ di) * (1-yd[i-1]) * (1+yd[i-1])
            dw.insert(0, di.copy())
        return dw

    # recalcula los pesos
    def __amend(self, ys: np.ndarray, delta: list):
        for i in range(self.__layers):
            y = self.__add_bias(ys[i])
            dw = self.__learn_coef * (np.transpose([delta[i]]) @ [y])
            self.__weights[i] += dw

    # funcion de entrenamiento por lotes
    def trn(self, x_in, d):
        for i in range(len(x_in)):
            ys = self.__propagation(x_in[i])
            dw = self.__back_propagation(ys, d[i])
            ys.insert(0, x_in[i].copy())
            self.__amend(ys, dw)

    # funcion para calcular el porcentaje de error cuadratico total
    def score(self, data_set: np.ndarray, y_d: np.ndarray, etype: str = "error_cuadratico") -> float:
        err_tot = 0
        for i in range(len(data_set)):
            if etype == "error_cuadratico":
                err_tot += sum((y_d[i] - self.eval(data_set[i]))**2)
            elif etype == "porcentaje_error":
                if np.all(np.equal(y_d[i], winner_take_all(self.eval(data_set[i])))): err_tot+=1
            else:
                raise ValueError(f"{etype} no es un tipo de error valido")
            
        err = err_tot/len(data_set)
        return err 

    def getWeigth(self) -> list:
        return self.__weights

    def __str__(self) -> str:
        return f"la arquitectura del modelo es: {self.__architecture}\ncon una profundidad de {len(self.__architecture)}"