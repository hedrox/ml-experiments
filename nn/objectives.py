"""
Defined loss functions for our custom neural network implementation
"""

import numpy as np


def cross_entropy(target: np.ndarray, output: np.ndarray) -> np.float64:
    """
    Computes cross entropy between target and output
    """
    return np.mean(-np.sum(target * np.log(output)))

def binary_cross_entropy(target: np.ndarray, output: np.ndarray) -> np.float64:
    """
    Computes binary cross entropy between target and output
    """
    return np.mean(-(target * np.log(output) + (1-target) * np.log(1-output)))

def MSE(target: np.ndarray, output: np.ndarray) -> np.float64:
    """
    Computes mean squared error between target and output
    """
    return np.mean((output - target)**2)

def MAE(target: np.ndarray, output: np.ndarray) -> np.float64:
    """
    Computes mean absolute error between target and output
    """
    return np.mean(np.abs(output - target))

def kl_divergence(target: np.ndarray, output: np.ndarray) -> np.float64:
    """
    Computes KL divergence between target and output
    """
    return np.sum(target * np.log(target / output), axis=-1)
