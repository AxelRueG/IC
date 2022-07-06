import numpy as np
from activation_funcs import sigmoide 


class MultiLayerPerceptron():
  def __init__(self,architecture,data,mu=0.1):
    self.architecture = architecture
    self.N = len(self.architecture)
    self.x = data[:,0:-1].copy()
    self.yd = data[:,-self.architecture[-1]::].copy()
    self.weights = []
    self.mu = mu

    # weights with bias as a input
    l_inputs = self.x.shape[1]+1
    for layer in self.architecture:
      # matrix m,n where m = amount of neurons and n = amount of inputs 
      self.weights.append(np.random.rand(layer,l_inputs)-0.5)
      l_inputs = layer+1

  def eval(self,x_in):
    y = []
    x = x_in.copy()
    for i in range(self.N):
      # the first element is the bias with value -1
      x = self.add_bias(x)
      y.append(sigmoide(np.dot(self.weights[i],x)))
      # the new input is the output of the last layer
      x = y[-1].copy()
    return y

  def backward_propagation(self,it,y):
    delta_w = []
    # output layer
    e = self.yd[it]-y[-1]  # error de evalucion
    delta_j = 0.5*e*(1+y[-1])*(1-y[-1])
    delta_w.append(np.transpose([self.mu*delta_j])*self.add_bias(y[-2]))
    # internal layers

    # input layer

    return delta_w
    

  def add_bias(self,x):
    return np.concatenate((np.array([-1]),x))

  def training(self,it):
    y = self.eval(self.x[it])
    delta_w = self.backward_propagation(it)

  def show_weight(self):
    for i in range(self.N):
      print(self.weights[i].shape)



if __name__=='__main__':
  MLP = MultiLayerPerceptron([4,3],np.array(
      [[ 4.5,  2.3,  1.3,  0.3, -1. , -1. ,  1. ],
       [ 5.1,  3.3,  1.7,  0.5, -1. , -1. ,  1. ],
       [ 7.2,  3. ,  5.8,  1.6,  1. , -1. , -1. ],
       [ 5.5,  4.2,  1.4,  0.2, -1. , -1. ,  1. ],
       [ 6.7,  3.1,  4.7,  1.5, -1. ,  1. , -1. ],
       [ 6.4,  3.1,  5.5,  1.8,  1. , -1. , -1. ],
       [ 6.1,  3. ,  4.9,  1.8,  1. , -1. , -1. ],
       [ 5.2,  3.4,  1.4,  0.2, -1. , -1. ,  1. ],
       [ 5. ,  3.3,  1.4,  0.2, -1. , -1. ,  1. ],
       [ 6.7,  3.3,  5.7,  2.1,  1. , -1. , -1. ]]))
  MLP.backward_propagation(0,[np.array([-0.32, 1.32,0.55,-0.22]),np.array([-0.98,-1.00001,0.998])])
  MLP.show_weight()
        

