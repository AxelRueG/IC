import numpy as np

# funcion signo
def sgn(x):
    return -1 if x < 0 else 1

def sigmoide(x):
    return (2/(1+np.exp(-x)))-1

def sgn_vec(x: np.ndarray) -> np.ndarray: 
    return (x/abs(x)).astype(int)