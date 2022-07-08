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

  def backward_propagation(self,it,ys):
    # add at the start of y's the input as y_0
    y = ys.copy()
    y.insert(0,self.add_bias(self.x[it]))
    delta = []
    # output layer
    e = self.yd[it]-y[-1]
    di = 0.5*e*(1+y[-1])*(1-y[-1])
    delta.append(di.copy())

    # internal layers
    for i in range(self.N,1,-1):
      di = 0.5*np.dot(np.transpose(self.weights[i-1][:,1::]),di)
      # *(1-y[i-2])*(1+y[i-2])
      delta.insert(0,di.copy())
    
    print(delta)

    return delta

  def training(self, it) :
    """
    update the weights of the MLP in base an iterator [it] of the inputs list
    """
    # calculate the outpus and the deltas for updates
    x = self.eval(self.x[it])
    delta = self.backward_propagation(it,x)

    # add the input to array of outputs for the loop
    x.insert(0,self.x[it]) 
    
    for i in range(0,self.N):
      xs = np.array([self.add_bias(x[i])])
      self.weights[i] -= self.mu*np.dot(np.transpose([delta[i]]),xs)
      print(self.weights[i])    

  def add_bias(self,x):
    return np.concatenate((np.array([-1]),x))

  def show_weight(self):
    for i in range(self.N):
      print(self.weights[i].shape)



if __name__=='__main__':
  MLP = MultiLayerPerceptron([3,2,1],np.array(
      [[ -1. , -1. , -1. ],
       [ -1. ,  1. ,  1. ],
       [  1. , -1. ,  1. ],
       [  1. ,  1. , -1. ]]))
  # MLP = MultiLayerPerceptron([4,3],np.array(
  #     [[ 4.5,  2.3,  1.3,  0.3, -1. , -1. ,  1. ],
  #      [ 5.1,  3.3,  1.7,  0.5, -1. , -1. ,  1. ],
  #      [ 7.2,  3. ,  5.8,  1.6,  1. , -1. , -1. ],
  #      [ 5.5,  4.2,  1.4,  0.2, -1. , -1. ,  1. ],
  #      [ 6.7,  3.1,  4.7,  1.5, -1. ,  1. , -1. ],
  #      [ 6.4,  3.1,  5.5,  1.8,  1. , -1. , -1. ],
  #      [ 6.1,  3. ,  4.9,  1.8,  1. , -1. , -1. ],
  #      [ 5.2,  3.4,  1.4,  0.2, -1. , -1. ,  1. ],
  #      [ 5. ,  3.3,  1.4,  0.2, -1. , -1. ,  1. ],
  #      [ 6.7,  3.3,  5.7,  2.1,  1. , -1. , -1. ]]))
  MLP.backward_propagation(0,[np.array([0.01,-0.32, 1.32]),np.array([-0.32, 1.32]),np.array([-0.98])])
  MLP.show_weight()
        

