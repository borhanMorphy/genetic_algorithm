import numpy as np

def relu(A):
    return np.maximum(0, A)

def softmax(A):
    e_x = np.exp(A - np.max(A))
    return e_x / e_x.sum(axis=0)