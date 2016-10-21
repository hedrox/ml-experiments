import numpy as np

def cross_entropy(target, output):
        return np.mean(-np.sum(target * np.log(output)))

def binary_cross_entropy(target, output):
    return np.mean(-(target * np.log(output) + (1-target) * np.log(1-output)))

def MSE(target, output):
    return np.mean((output - target)**2)

def MAE(target, output):
    return np.mean(np.abs(output - target))

def KL_divergence(target, output):
    return np.sum(target * np.log(target / output), axis=-1)