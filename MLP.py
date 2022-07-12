import numpy as np
from activation_funcs import sigmoide, signo 


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
    # y.insert(0,self.add_bias(self.x[it]))
    delta = []
    # output layer
    e = self.yd[it]-y[-1]
    di = 0.5*e*(1+y[-1])*(1-y[-1])
    delta.append(di.copy())

    # internal layers
    for i in range(self.N-1,0,-1):
      di = 0.5*np.dot(np.transpose(self.weights[i][:,1::]),di)*(1-y[i-1])*(1+y[i-1])
      delta.insert(0,di.copy())

    return delta

  def training(self, it) :
    # calculate the outpus and the deltas for updates
    x = self.eval(self.x[it])
    delta = self.backward_propagation(it,x)
    # add the input to array of outputs for the loop
    x.insert(0,self.x[it]) 

    for i in range(0,self.N):
      xs = np.array([self.add_bias(x[i])])
      delta_weights = self.mu*np.dot(np.transpose([delta[i]]),xs)
      print(self.weights[i].shape,delta_weights.shape)
      self.weights[i] -= delta_weights

  def training_epoc(self):
    # training
    for i in range(self.x.shape[0]): self.training(i)
    # eval error in epoc
    error_in_epoc = 0
    for i in range(self.x.shape[0]):
      y = self.eval(self.x[i])[-1]
      # if np.all(signo(y) != self.yd[i]): error_in_epoc+=1
      # mean absolute error
      error_in_epoc += np.abs(self.yd[i]-y)
    return error_in_epoc/self.x.shape[0]

  def test(self,data):
    # eval error in epoc
    x = data[:,0:-1].copy()
    d = data[:,-self.architecture[-1]::].copy()
    error_in_epoc = 0
    for i in range(data.shape[0]):
      y = self.eval(x[i])[-1]
      # if np.all(signo(y) != d[i]): error_in_epoc+=1
      error_in_epoc += np.abs((d[i]-y))
    return error_in_epoc/data.shape[0]

  def add_bias(self,x):
    return np.concatenate((np.array([-1]),x))

  def show_weight(self):
    for i in range(self.N):
      print(self.weights[i].shape)

