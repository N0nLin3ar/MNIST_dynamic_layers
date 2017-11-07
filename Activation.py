import numpy as np


# Performs the specific activation function on (z)
def sigmoid(z):
    s = 1 / (1 + np.exp(-z))

    return s

def tanh(z):
    s = (np.exp(z)-np.exp(-z)) / (np.exp(z)+np.exp(-z))

    return s

def relu(z):
    s = np.maximum(z, 0, z) # np.maximum(z, 0, z) is more efficient than np.maximum(0, z)

    return s