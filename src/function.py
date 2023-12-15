import numpy as np

def sigmoid(x, x0, k, L, b):
    y = L / (1 + np.exp(-k*(x-x0))) + b
    return (y)