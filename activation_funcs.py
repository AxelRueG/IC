import numpy as np

def sigmoide(z):
  return 2/(1+np.exp(-z))-1

def escalon(x):
  return -1 if x<0 else 1

def signo(x):
  xs = np.ones(x.shape[0])
  index = np.nonzero(x<0)[0]
  xs[index] = -1
  return xs
