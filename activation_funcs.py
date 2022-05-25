import numpy as np

def sigmoide(z):
  x = np.exp(-z)
  return (1-x)/(1+x)

def escalon(x):
  return -1 if x<0 else 1