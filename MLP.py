import numpy as np 


class MultiLayerPerceptron():
    def __init__(self,architecture,data):
        self.architecture = architecture
        self.N = len(self.architecture)
        self.x = data[:,0:-1].copy()
        self.yd = data[:,-1].copy()
        self.weights = []

        # weights with bias as a input
        l_inputs = self.x.shape[0]+1
        for layer in self.architecture:
            # matrix m,n where m = amount of neurons and n = amount of inputs 
            self.weights.append(np.random.rand(layer,l_inputs)-0.5)
            l_inputs = layer+1
        
    def eval(self,x_in):
        y = []
        x = x_in.copy()
        for i in range(self.N):
            # the first element is the bias with value -1
            x = np.concatenate((np.array([-1]),x))
            y.append(sigmoide(np.dot(self.weights[i],x)))
            # the new input is the output of the last layer
            x = y[-1].copy()
        return y
        
    def backward_propagation(self,it):
        # capa de salida
        pass

    def training(self,it):
        y = self.eval(self.x[it])
        delta_w = self.backward_propagation(it)
        



        

